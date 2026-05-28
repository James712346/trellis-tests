function [best_path, log_prob] = hmm_viterbi(obs_seq, log_initial, log_emission, log_transition)
% HMM_VITERBI  Most-likely state sequence — GPU accelerated.
%
% All table inputs may be CPU or already gpuArray.
% Pre-uploading with gpuArray() before calling in a loop is much faster.
%
% Inputs:
%   obs_seq       - (1 x T) or (T x 1) pixel intensities in [0,255]
%   log_initial   - (S x 1) single
%   log_emission  - (256 x S) single
%   log_transition- (S x S) single, dense, gpuArray preferred
%
% Outputs:
%   best_path - (1 x T) state indices (1-based), on CPU
%   log_prob  - scalar, on CPU

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

    % Viterbi forward pass
    g_delta = log_initial + log_emission(obs_seq(1)+1, :)';    % S x 1

    % Backpointers stored on CPU (uint32) — written once per step, read during backtrack
    psi = zeros(S, T, 'uint32');

    for t = 2:T
        scores            = bsxfun(@plus, g_delta, log_transition);   % S x S
        [best_vals, best_idx] = max(scores, [], 1);                    % 1 x S each
        g_delta           = best_vals' + log_emission(obs_seq(t)+1, :)';
        psi(:, t)         = uint32(gather(best_idx)');
    end

    % Backtrack on CPU
    [log_prob, last] = max(gather(g_delta));
    best_path        = zeros(1, T, 'uint32');
    best_path(T)     = uint32(last);
    for t = T-1:-1:1
        best_path(t) = psi(best_path(t+1), t+1);
    end
    best_path = double(best_path);
end
