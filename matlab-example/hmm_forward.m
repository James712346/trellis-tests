function log_prob = hmm_forward(obs_seq, log_initial, log_emission, log_transition)
% HMM_FORWARD  Log-probability of an observation sequence.
%
% All table inputs may be CPU (single/double) or already gpuArray.
% If CPU, they are uploaded here. For best performance when processing
% many sequences, pre-upload with gpuArray() and pass directly.
%
% Inputs:
%   obs_seq       - (1 x T) or (T x 1) pixel intensities in [0,255]
%   log_initial   - (S x 1) single
%   log_emission  - (256 x S) single
%   log_transition- (S x S) single, dense, gpuArray preferred
%
% Output:
%   log_prob - scalar (gathered to CPU)

    obs_seq = uint16(obs_seq(:)');
    T = numel(obs_seq);
    S = numel(log_initial);

    % Upload to GPU if not already there
    if ~isa(log_initial,   'gpuArray'), log_initial   = gpuArray(log_initial);   end
    if ~isa(log_emission,  'gpuArray'), log_emission  = gpuArray(log_emission);  end
    if ~isa(log_transition,'gpuArray')
        log_transition = gpuArray(single(full(log_transition)));
        log_transition(isinf(log_transition) & log_transition < 0) = single(-1e30);
    end

    % Forward algorithm
    g_alpha = log_initial + log_emission(obs_seq(1)+1, :)';

    for t = 2:T
        incoming = bsxfun(@plus, g_alpha, log_transition);    % S x S
        m        = max(incoming, [], 1);                       % 1 x S
        g_alpha  = m' + log(sum(exp(bsxfun(@minus, incoming, m)), 1))' ...
                   + log_emission(obs_seq(t)+1, :)';
    end

    m        = max(g_alpha);
    log_prob = gather(m + log(sum(exp(g_alpha - m))));
end
