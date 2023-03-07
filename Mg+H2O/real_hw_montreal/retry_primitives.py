import signal, time

from qiskit_ibm_runtime import Sampler, Estimator, Session
from qiskit.providers import JobStatus

def timeout_handler(signum, frame):
    raise Exception('Iteration timed out')

class RetryPrimitiveMixin:
    """RetryPrimitive class.
    
    This class inherits from Qiskit IBM Runtime's Primitives and overwrites its run method such that it retries calling it
    a maximum of 'max_retries' consecutive times, if it encounters one of the following randomly occuring errors:
    
    * A Primitive error (in this case "Job.ERROR" is printed, and the job is cancelled automatically)
    * A timeout error where the job either remains running or completes but does not return anything, for a time larger 
      than 'timeout' (in this case the job is cancelled by the patch and "Job.CANCELLED" is printed)
    * A creation error, where the job fails to be created because connection is lost between the runtime server and the
      quantum computer (in this case "Failed to create job." is printed). If this error occurs, the patch connects the user
      to a new Session (to be handled with care! also, this will unfortunately put the next job in the queue). 
    """
    
    def __init__(self, *args, max_retries: int = 5, timeout: int = 3600, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.timeout = timeout
        self.backend = super().session._backend
        signal.signal(signal.SIGALRM, timeout_handler)
        
    def run(self, *args, **kwargs):
        result = None
        for i in range(self.max_retries):
            try:
                job = super().run(*args, **kwargs)
                while job.status() in [JobStatus.INITIALIZING, JobStatus.QUEUED, JobStatus.VALIDATING]:
                    time.sleep(5) # Check every 5 seconds whether job status has changed
                signal.alarm(self.timeout) # Once job starts running, set timeout to 1 hour by default
                result = job.result()
                if result is not None:
                    signal.alarm(0) # Reset timer
                    return job
            except Exception as e:
                signal.alarm(0) # Reset timer
                print("\nSomething went wrong...")
                print(f"\n\nERROR MESSAGE:\n{e}\n\n")
                if 'job' in locals(): # Sometimes job fails to create
                    print(f"Job ID: {job.job_id}. Job status: {job.status()}.")
                    if job.status() not in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]:
                        job.cancel()
                else:
                    print("Failed to create job.")
                    try:
                        super().session.close()
                        print("Current session was closed.")
                    except:
                        print("Current session could not be closed. Will leave it to close automatically.")
                    print(f"Creating new session...\n")
                    self._session = Session(backend=self.backend)
                print(f"Starting trial number {i+2}...\n")
                signal.alarm(0) # Reset timer
        if result is None:
            raise RuntimeError(f"Program failed! Maximum number of retries ({self.max_retries}) exceeded")
            
class RetrySampler(RetryPrimitiveMixin, Sampler):
    pass

class RetryEstimator(RetryPrimitiveMixin, Estimator):
    pass