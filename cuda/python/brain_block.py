class BrainBlock:
    def __init__(self, property, idx, weight, bid=0, gid=0, delta_t=1):
        # property shape is [N, 22]
        # idx shape is [4, X]
        # weight shape is [X]
        # where (idx, weight) define a
        # sparse matrix with shape of [K, N, K, 4]
        # the sort of idx columns is the lexsort of
        # ([idx[0], idx[1], idx[2], idx[3]) (input driven, will be
        # departed in the future), and in future, is
        # ([idx[0], idx[1], idx[2], idx[3])(output driven)
        # bid: block_id, gid: gpu_id, delta_t float.
        pass

    def run(self, noise_rate, iter=1, I_ext=None, need_state=False):
        # running one step with the delta_t.
        # noise_rate = noise_freq / (1000/delta_t)
        pass

    def update_property(self, property):
        # update property
        pass

    def update_property(self, property_idx, property_weight):
        # update property
        pass

    def update_conn_weight(self, conn_weight):
        # update part of conn_weight
        # conn_idx shape is [4, X']
        # conn_weight shape is [X']
        pass

    def update_conn_weight(self, conn_idx, conn_weight):
        # update part of conn_weight
        # conn_idx shape is [4, X']
        # conn_weight shape is [X']
        pass

    def get_freqs(self):
        # get the sum activates of each block.
        # if block number is 90.
        # then F shape is [90]
        return F

    def get_vmeans(self):
        # get the mean voltage of membranes of each block.
        # if block number is 90.
        # then V shape is [90]
        return V

    def set_sample_idx(self, idx):
        # set the idx of the sample neurons.
        return True #if set success, if in batch_sample_status, reset sample idx is not allowed.

    def sample_for_show(self):
        # get the output spike and v_membranes for each sample neurons.
        return A, v_i

    def sample_for_debug(self):
        # get the J_ui, input spike for each sample neurons.
        return A_in, J_ui

    def sample_for_parameter_check(self):
        # get the property, idx, weight for the sample neurons.
        return property, idx, weight

    def status(self):
        # get the hardware status, e.g. GPU/CPU/FPGA/communicate loads, speed, reduction ratio.
        # different hardware platform can reuturn differnt status, as their willing respectively.
        return status