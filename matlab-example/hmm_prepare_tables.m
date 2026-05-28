function [log_init, log_emit, log_trans_sparse] = hmm_prepare_tables(tables, config)
% HMM_PREPARE_TABLES  Flatten and log-transform probability tables.
%
% Returns CPU-side sparse log_trans — caller is responsible for sending
% to GPU via hmm_forward / hmm_viterbi (which call gpuArray internally).
%
% Keeping sparse here avoids OOM during table prep for large configs.

    b  = config.num_buckets;
    t  = config.num_textures;
    np = config.grid_width * config.grid_height;
    S  = np * b * t;

    fprintf('  State space: %d states  (%dx%d grid, %d buckets, %d textures)\n', S, ...
        config.grid_width, config.grid_height, b, t);

    %% --- Initial probabilities (S x 1, single) ---
    init_flat = single(reshape(tables.initialProb, S, 1));
    init_flat = max(init_flat, single(1e-30));
    log_init  = log(init_flat);

    %% --- Emission probabilities (256 x S, single) ---
    emit_full = zeros(256, S, 'single');
    for s = 0:S-1
        bkt = mod(floor(s / t), b);
        emit_full(:, s+1) = single(tables.emissionProb(:, bkt+1));
    end
    emit_full = max(emit_full, single(1e-30));
    log_emit  = log(emit_full);

    %% --- Transition matrix (sparse, log of non-zeros only) ---
    % We keep this sparse on CPU. Zeros stay implicit (-inf in log space).
    % This avoids the memory explosion of log(full dense matrix).
    fprintf('  Building sparse transition matrix ...\n');
    build_start = tic;

    [sp_rows, sp_cols, sp_vals] = find(tables.spatialWeight);
    nnz_spatial = numel(sp_vals);

    max_nnz = nnz_spatial * b * t * b * t;
    tr_rows = zeros(max_nnz, 1, 'uint32');
    tr_cols = zeros(max_nnz, 1, 'uint32');
    tr_vals = zeros(max_nnz, 1, 'single');
    tr_idx  = 0;

    for sp_i = 1:nnz_spatial
        p1 = sp_rows(sp_i) - 1;
        p2 = sp_cols(sp_i) - 1;
        sw = single(sp_vals(sp_i));

        for b1 = 0:b-1
            for b2 = 0:b-1
                iw = single(tables.intensityWeight(b1+1, b2+1));
                if iw == 0, continue; end

                for t1 = 0:t-1
                    for t2 = 0:t-1
                        tw = single(tables.textureWeight(t1+1, t2+1));
                        if tw == 0, continue; end

                        prob = sw * iw * tw ...
                             * single(tables.positionBias(p2+1)) ...
                             * single(tables.bucketBias(b2+1))   ...
                             * single(tables.textureBias(t2+1));

                        if prob == 0, continue; end

                        s1 = uint32(p1*(b*t) + b1*t + t1 + 1);
                        s2 = uint32(p2*(b*t) + b2*t + t2 + 1);

                        tr_idx = tr_idx + 1;
                        tr_rows(tr_idx) = s1;
                        tr_cols(tr_idx) = s2;
                        tr_vals(tr_idx) = prob;
                    end
                end
            end
        end
    end

    % Build sparse matrix, row-normalise, then log only the non-zeros
    trans_sparse = sparse(double(tr_rows(1:tr_idx)), double(tr_cols(1:tr_idx)), ...
                          double(tr_vals(1:tr_idx)), S, S);

    row_sums = full(sum(trans_sparse, 2));
    row_sums(row_sums == 0) = 1;
    norm_diag = spdiags(1./row_sums, 0, S, S);
    trans_sparse = norm_diag * trans_sparse;

    % Log only the stored non-zeros — zeros remain implicit (-inf)
    [r, c, v] = find(trans_sparse);
    v = max(single(v), single(1e-30));
    log_trans_sparse = sparse(r, c, double(log(v)), S, S);

    fprintf('  Transition matrix ready: %.2f s  nnz=%d  sparsity=%.4f%%\n', ...
        toc(build_start), nnz(log_trans_sparse), ...
        (1 - nnz(log_trans_sparse)/S^2)*100);
end
