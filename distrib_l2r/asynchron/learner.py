import logging
import queue
import random
import socketserver
import threading
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
import time
from tqdm import tqdm
import sys
import socket
from src.agents.base import BaseAgent
from src.config.yamlize import create_configurable, NameToSourcePath, yamlize
from src.loggers.WanDBLogger import WanDBLogger


from distrib_l2r.api import BufferMsg
from distrib_l2r.api import InitMsg
from distrib_l2r.api import EvalResultsMsg
from distrib_l2r.api import PolicyMsg
from distrib_l2r.utils import receive_data
from distrib_l2r.utils import send_data


class ThreadPoolMixin:
    """
    Mixin to use a fixed pool of threads to handle requests.
    .. note::
       When shutting down the server, please ensure you call this mixin's
       `join()` to shut down the pool along with the server's `shutdown()`
       method. The order in which these are performed is not significant,
       but both actions must be performed.
    """

    # Size of pool.
    pool_size = 5

    # How long to wait on an empty queue, in seconds. Can be a float.
    timeout_on_get = 0.5

    def __init__(self):
        self._request_queue = queue.Queue(self.pool_size)
        # This beastie serves a different purpose than __shutdown_request
        # and __is_shut_down: those are superprivate so we can't touch them,
        # and even if we could, they're not really useful in shutting down
        # the pool.
        self._shutdown_event = threading.Event()
        for _ in range(self.pool_size):
            thread = threading.Thread(target=self.process_request_thread)
            thread.setDaemon(1)
            thread.start()

    def process_request_thread(self):
        """Same as in BaseServer, but as a thread."""
        while True:
            try:
                request, client_address = self._request_queue.get(
                    timeout=self.timeout_on_get,
                )
            except queue.Empty:
                if self._shutdown_event.isSet():
                    return
                continue
            try:
                self.finish_request(request, client_address)
                self.shutdown_request(request)
            except:
                self.handle_error(request, client_address)
                self.shutdown_request(request)
            self._request_queue.task_done()

    def process_request(self, request, client_address):
        """Queue the given request."""
        self._request_queue.put((request, client_address))

    def join(self):
        """Wait on the pool to clear and shut down the worker threads."""
        # A nicer place for this would be shutdown(), but this being a mixin,
        # that method can't safely do anything with that method, thus we add
        # an extra method explicitly for clearing the queue and shutting
        # down the workers.
        self._request_queue.join()
        self._shutdown_event.set()

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    """Request handler thread created for every request"""

    def handle(self) -> None:
        """ReplayBuffers are not thread safe - pass data via thread-safe queues"""
        msg = receive_data(self.request)

        # Received a replay buffer from a worker
        # Add this to buff
        if isinstance(msg, BufferMsg):
            self.server.buffer_queue.put(msg.data)

        # Received an init message from a worker
        # Immediately reply with the most up-to-date policy
        elif isinstance(msg, InitMsg):
            logging.info("Received init message")

        # Received evaluation results from a worker
        elif isinstance(msg, EvalResultsMsg):
            self.server.wandb_logger.log_metric(
                
                    msg.data["reward"], 'reward'
                
            )

        # unexpected
        else:
            logging.warning(f"Received unexpected data: {type(msg)}")
            return

        # Reply to the request with an up-to-date policy
        send_data(data=PolicyMsg(data=self.server.get_agent_dict()), sock=self.request)




class AsyncLearningNode(ThreadPoolMixin, socketserver.TCPServer):
    """A multi-threaded, offline, off-policy reinforcement learning server

    Args:
        policy: an intial Tianshou policy
        update_steps: the number of gradient updates for each buffer received
        batch_size: the batch size for gradient updates
        epochs: the number of buffers to receive before concluding learning
        server_address: the address the server runs on
        eval_freq: the likelihood of responding to a worker to eval instead of train
        save_func: a function for saving which is called while learning with
          parameters `epoch` and `policy`
        save_freq: the frequency, in epochs, to save
    """

    def __init__(
        self,
        agent: BaseAgent,
        update_steps: int = 300,
        batch_size: int = 128, # Originally 128
        epochs: int = 2, # Originally 500
        buffer_size: int = 1_000_000, # Originally 1M
        server_address: Tuple[str, int] = ("0.0.0.0", 4444),
        eval_prob: float = 0.20,
        save_func: Optional[Callable] = None,
        save_freq: Optional[int] = None,
        api_key: str = "",
    ) -> None:

        super().__init__(server_address, ThreadedTCPRequestHandler)

        self.update_steps = update_steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_prob = eval_prob

        # Create a replay buffer
        self.buffer_size = buffer_size
        self.replay_buffer = create_configurable(
            "config_files/async_sac_mountaincar/buffer.yaml", NameToSourcePath.buffer
        )

        # Inital policy to use
        self.agent = agent
        self.agent_id = 1

        # The bytes of the policy to reply to requests with

        self.updated_agent = {k: v.cpu() for k, v in self.agent.state_dict().items()}

        # A thread-safe policy queue to avoid blocking while learning. This marginally
        # increases off-policy error in order to improve throughput.
        self.agent_queue = queue.Queue(maxsize=1)

        # A queue of buffers that have been received but not yet added to the learner's
        # main replay buffer
        self.buffer_queue = queue.LifoQueue()

        self.wandb_logger = WanDBLogger(api_key=api_key, project_name="test-project")
        # Save function, called optionally
        self.save_func = save_func
        self.save_freq = save_freq

    def get_agent_dict(self) -> Dict[str, Any]:
        """Get the most up-to-date version of the policy without blocking"""
        if not self.agent_queue.empty():
            try:
                self.updated_agent = self.agent_queue.get_nowait()
            except queue.Empty:
                # non-blocking
                pass

        return {
            "policy_id": self.agent_id,
            "policy": self.updated_agent,
            "is_train": random.random() >= self.eval_prob,
        }

    def update_agent(self) -> None:
        """Update policy that will be sent to workers without blocking"""
        if not self.agent_queue.empty():
            try:
                # empty queue for safe put()
                _ = self.agent_queue.get_nowait()
            except queue.Empty:
                pass

        self.agent_queue.put({k: v.cpu() for k, v in self.agent.state_dict().items()})
        self.agent_id += 1

    
    def learn(self) -> None:
        """The thread where thread-safe gradient updates occur"""
        for epoch in range(self.epochs):
            print("Waiting on queue")
            semibuffer = self.buffer_queue.get()
            print(f"Received something {len(semibuffer)} vs {len(self.replay_buffer)}. {self.buffer_queue.qsize()} buffers remaining")
            # Add new data to the primary replay buffer
            self.replay_buffer.store(semibuffer)

            print("Training!")
            # Learning steps for the policy
            for _ in tqdm(range(max(1,min(self.update_steps, len(self.replay_buffer) // self.replay_buffer.batch_size)))):
                batch = self.replay_buffer.sample_batch()
                self.agent.update(data=batch)
                #print(next(self.agent.actor_critic.policy.mu_layer.parameters()))

            print("Trying to update agent queue")
            # Update policy without blocking
            self.update_agent()
            # Optionally save
            if self.save_func and epoch % self.save_every == 0:
                self.save_fn(epoch=epoch, policy=self.get_policy_dict())

    def server_bind(self):
        # From https://stackoverflow.com/questions/6380057/python-binding-socket-address-already-in-use/18858817#18858817.
        # Tries to ensure reuse. Might be wrong.
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)
