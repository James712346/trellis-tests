function results = runner(varargin)
% RUNNER  Full pipeline: load/generate tables, run HMM on images, log results.
%
% Usage:
%   results = runner('SMALL')
%   results = runner('SMALL', 'MEDIUM')
%   results = runner('SMALL', 'mode', 'forward')
%   results = runner('SMALL', 'mode', 'viterbi')
%   results = runner('SMALL', 'mode', 'both')           % default
%   results = runner('SMALL', 'images', {'my.jpg'})
%   results = runner('SMALL', 'method', 'cols')         % process columns instead of rows
%   results = runner('SMALL', 'max_sequences', 50)
%   results = runner('SMALL', 'regenerate_tables', true) % force table rebuild
%   results = runner('SMALL', 'no_warmup', true)
%
% Multiple config names can be listed before any name-value pairs, e.g.:
%   results = runner('SMALL', 'MEDIUM', 'LARGE', 'mode', 'forward')
%
% Output:
%   results - struct array, one element per config, with fields:
%       config_name, mode, run_uuid, start_time, end_time, duration_s,
%       image_results (struct array per image), status, error_message

    %% --- Parse arguments ---
    p = inputParser();
    p.KeepUnmatched = false;

    % Collect leading positional config names
    config_names = {};
    i = 1;
    while i <= numel(varargin)
        v = varargin{i};
        if ischar(v) && ~contains(v, '=') && (i == numel(varargin) || ischar(varargin{i+1}) && ismember(upper(v), fieldnames(get_configs())))
            % Looks like a config name
            all_cfg = fieldnames(get_configs());
            if ismember(upper(v), all_cfg)
                config_names{end+1} = upper(v); %#ok<AGROW>
                i = i + 1;
                continue;
            end
        end
        break;
    end
    remaining = varargin(i:end);

    addParameter(p, 'mode',               'both',  @(x) ismember(x, {'forward','viterbi','both'}));
    addParameter(p, 'images',             {},      @iscell);
    addParameter(p, 'method',             'rows',  @(x) ismember(x, {'rows','cols'}));
    addParameter(p, 'max_sequences',      inf,     @isnumeric);
    addParameter(p, 'regenerate_tables',  false,   @islogical);
    addParameter(p, 'no_warmup',          false,   @islogical);
    addParameter(p, 'quiet',              false,   @islogical);
    parse(p, remaining{:});
    opts = p.Results;

    if isempty(config_names)
        error('runner: supply at least one config name, e.g. runner(''SMALL'')');
    end


    %% --- GPU check (hard requirement) ---
    if ~parallel.gpu.GPUDevice.isAvailable()
        error('runner: no CUDA GPU available. This pipeline requires a GPU.');
    end
    gpu_dev = gpuDevice();
    fprintf('GPU: %s  |  VRAM: %.1f GB total, %.1f GB free\n\n', ...
        gpu_dev.Name, gpu_dev.TotalMemory/1e9, gpu_dev.AvailableMemory/1e9);

    do_warmup = ~opts.no_warmup;
    configs   = get_configs();

    % Pre-define results with all expected fields so appending never mismatches
    empty_run = struct('config_name','','mode','','run_uuid','', ...
        'start_time',datetime('now'),'end_time',datetime('now'), ...
        'duration_s',0,'status','','error_message','','image_results',[]);
    results = repmat(empty_run, 0, 1);

    %% --- Loop over configs ---
    for ci = 1:numel(config_names)
        cfg_name = config_names{ci};
        if ~isfield(configs, cfg_name)
            warning('runner: unknown config "%s", skipping.', cfg_name);
            continue;
        end
        if ~opts.quiet && numel(config_names) > 1
            fprintf('[Config %d/%d] %s\n', ci, numel(config_names), cfg_name);
        end
        config = configs.(cfg_name);

        run_result.config_name = cfg_name;
        run_result.mode        = opts.mode;
        run_result.run_uuid    = generate_uuid();
        run_result.start_time  = datetime('now', 'TimeZone', 'local');
        run_result.status      = 'running';
        run_result.error_message = '';
        run_result.image_results = {};

        banner(sprintf('CONFIGURATION: %s', cfg_name), opts.quiet);

        try
            %% --- Load or generate tables ---
            tables_dir = fullfile('models', lower(cfg_name), 'tables');
            tables = load_or_generate_tables(cfg_name, config, tables_dir, opts.regenerate_tables, opts.quiet);

            %% --- Prepare log-domain HMM tables ---
            if ~opts.quiet
                fprintf('Preparing HMM log-tables ...\n');
            end
            t_prep = tic;
            [log_init, log_emit, log_trans] = hmm_prepare_tables(tables, config);
            if ~opts.quiet
                fprintf('  Done in %.2f s\n\n', toc(t_prep));
            end

            %% --- Warmup run ---
            if do_warmup
                run_warmup(log_init, log_emit, log_trans, opts.mode, opts.quiet);
            end

            %% --- Find images ---
            image_list = find_images(opts.images, opts.quiet);

            if isempty(image_list)
                if ~opts.quiet
                    fprintf('No images found — generating synthetic sequences.\n');
                end
                rng(12345);
                synthetic = uint8(randi([0 255], 100, 200));
                image_list = {struct('name','synthetic','path','','data',synthetic)};
            end

            %% --- Run on each image ---
            for img_i = 1:numel(image_list)
                img_info = image_list{img_i};
                img_result = run_on_image(img_info, log_init, log_emit, log_trans, ...
                                          opts.mode, opts.method, opts.max_sequences, opts.quiet);
                if isempty(run_result.image_results)
                    run_result.image_results = img_result;
                else
                    run_result.image_results(end+1) = img_result;
                end
            end

            run_result.status = 'completed';

        catch ME
            run_result.status        = 'error';
            run_result.error_message = ME.message;
            fprintf('ERROR in config %s: %s\n', cfg_name, ME.message);
            fprintf('%s\n', getReport(ME));
        end

        run_result.end_time   = datetime('now', 'TimeZone', 'local');
        run_result.duration_s = seconds(run_result.end_time - run_result.start_time);

        %% --- Save results for this config ---
        save_results(run_result, cfg_name);

        if ~opts.quiet
            print_run_summary(run_result);
        end

        results(end+1) = run_result; %#ok<AGROW>
    end

    %% --- Final summary ---
    if ~opts.quiet && numel(results) > 1
        banner('OVERALL SUMMARY', false);
        for ri = 1:numel(results)
            r = results(ri);
            fprintf('  %-10s  status=%-12s  duration=%.2f s\n', ...
                r.config_name, r.status, r.duration_s);
        end
        fprintf('\n');
    end
end


% =========================================================================
%  Pipeline helpers
% =========================================================================

function tables = load_or_generate_tables(cfg_name, config, tables_dir, force, quiet)
% Load .mat tables from disk, or generate them if missing / forced.
    meta_path = fullfile(tables_dir, 'metadata.mat');

    if ~force && exist(meta_path, 'file')
        if ~quiet
            fprintf('Loading existing tables from %s ...\n', tables_dir);
        end
        tables = load_tables_from_dir(tables_dir);
    else
        if ~quiet
            fprintf('Generating tables for %s ...\n', cfg_name);
        end
        tables = generate_tables(cfg_name, tables_dir);
    end
end


function tables = load_tables_from_dir(tables_dir)
% Load all table .mat files from a directory into a struct.
    table_names = {'initialProb','emissionProb','spatialWeight', ...
                   'intensityWeight','textureWeight','positionBias', ...
                   'bucketBias','textureBias'};
    tables = struct();
    for i = 1:numel(table_names)
        fname = table_names{i};
        fpath = fullfile(tables_dir, [fname '.mat']);
        if ~exist(fpath, 'file')
            error('Missing table file: %s', fpath);
        end
        loaded = load(fpath, 'data');
        tables.(fname) = loaded.data;
    end
end


function run_warmup(log_init, log_emit, log_trans, mode, quiet)
% Run a short synthetic sequence to warm up the JIT / memory allocation.
    if ~quiet, fprintf('Running warmup ...\n'); end
    rng(0);
    warmup_seq = randi([0 255], 1, 20);
    if ismember(mode, {'forward','both'})
        hmm_forward(warmup_seq, log_init, log_emit, log_trans);
    end
    if ismember(mode, {'viterbi','both'})
        hmm_viterbi(warmup_seq, log_init, log_emit, log_trans);
    end
    if ~quiet, fprintf('  Warmup done.\n\n'); end
end


function img_result = run_on_image(img_info, log_init, log_emit, log_trans, ...
                                    mode, method, max_seq, quiet)
% Run forward/viterbi on all sequences extracted from one image.
    img_result.name      = img_info.name;
    img_result.path      = img_info.path;
    img_result.sequences = struct([]);

    % Load or use pre-supplied data
    if isfield(img_info, 'data')
        img = img_info.data;
    else
        if ~quiet, fprintf('  Loading: %s\n', img_info.path); end
        raw = imread(img_info.path);
        if size(raw, 3) == 3
            img = rgb2gray(raw);
        else
            img = raw;
        end
    end

    img_result.width  = size(img, 2);
    img_result.height = size(img, 1);

    % Extract sequences
    if strcmp(method, 'rows')
        sequences = mat2cell(double(img), ones(1,size(img,1)), size(img,2));
        n_seq = size(img, 1);
    else  % cols
        sequences = mat2cell(double(img)', ones(1,size(img,2)), size(img,1));
        n_seq = size(img, 2);
    end

    n_seq = min(n_seq, max_seq);
    if ~quiet
        fprintf('  Image: %s  [%dx%d]  %d sequences (%s)\n', ...
            img_info.name, img_result.height, img_result.width, n_seq, method);
    end

    % --- Pre-transfer tables to GPU once per image batch ---
    % Time separately so compute-only time is comparable to Trellis.
    if ~quiet, fprintf('  Transferring tables to GPU ...\n'); end
    t_transfer  = tic;
    g_log_init  = gpuArray(log_init);
    g_log_emit  = gpuArray(log_emit);
    g_log_trans = gpuArray(single(full(log_trans)));
    g_log_trans(isinf(g_log_trans) & g_log_trans < 0) = single(-1e30);
    wait(gpuDevice());
    transfer_time_s = toc(t_transfer);
    if ~quiet, fprintf('  GPU transfer: %.3f s\n', transfer_time_s); end

    t_img          = tic;
    forward_times  = zeros(1, n_seq);
    viterbi_times  = zeros(1, n_seq);
    log_probs_fwd  = zeros(1, n_seq);
    log_probs_vit  = zeros(1, n_seq);

    for si = 1:n_seq
        obs = uint16(sequences{si});

        if ismember(mode, {'forward','both'})
            tf = tic;
            log_probs_fwd(si) = hmm_forward(obs, g_log_init, g_log_emit, g_log_trans);
            wait(gpuDevice());
            forward_times(si) = toc(tf);
        end

        if ismember(mode, {'viterbi','both'})
            tv = tic;
            [~, log_probs_vit(si)] = hmm_viterbi(obs, g_log_init, g_log_emit, g_log_trans);
            wait(gpuDevice());
            viterbi_times(si) = toc(tv);
        end
        progress_bar(si, n_seq, toc(t_img), mode, quiet);
    end

    img_result.n_sequences     = n_seq;
    img_result.duration_s      = toc(t_img);
    img_result.transfer_time_s = transfer_time_s;
    img_result.log_probs_fwd   = log_probs_fwd;
    img_result.log_probs_vit   = log_probs_vit;
    img_result.forward_times   = forward_times;
    img_result.viterbi_times   = viterbi_times;

    if ismember(mode, {'forward','both'}) && ~quiet
        fwd_active = forward_times(forward_times > 0);
        fprintf('    Forward : mean=%.4f s/seq  total=%.2f s  seq/s=%.10f  mean log_prob=%.2f\n', ...
            mean(fwd_active), sum(fwd_active), numel(fwd_active)/sum(fwd_active), mean(log_probs_fwd));
    end
    if ismember(mode, {'viterbi','both'}) && ~quiet
        vit_active = viterbi_times(viterbi_times > 0);
        fprintf('    Viterbi : mean=%.4f s/seq  total=%.2f s  seq/s=%.10f  mean log_prob=%.2f\n', ...
            mean(vit_active), sum(vit_active), numel(vit_active)/sum(vit_active), mean(log_probs_vit));
    end
    if ~quiet
        fprintf('    (GPU transfer time excluded from above: %.3f s)\n', transfer_time_s);
    end
end


function image_list = find_images(filter_list, quiet)
% Search current directory (and subdirs) for image files.
    extensions = {'*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'};
    found = {};

    if ~isempty(filter_list)
        % Explicit list — resolve paths
        for i = 1:numel(filter_list)
            p = filter_list{i};
            if exist(p, 'file')
                [~, nm, ext] = fileparts(p);
                found{end+1} = struct('name', [nm ext], 'path', p); %#ok<AGROW>
            else
                warning('runner: image not found: %s', p);
            end
        end
    else
        % Auto-discover
        for ei = 1:numel(extensions)
            files = dir(extensions{ei});
            for fi = 1:numel(files)
                found{end+1} = struct('name', files(fi).name, 'path', files(fi).name); %#ok<AGROW>
            end
        end
    end

    image_list = found;
    if ~quiet
        fprintf('Found %d image(s).\n', numel(image_list));
    end
end


function save_results(run_result, cfg_name)
% Save run results to a .mat file and a human-readable text summary.
    out_dir = fullfile('results', lower(cfg_name));
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    ts  = char(run_result.start_time, 'yyyyMMdd_HHmmss');
    mat_path = fullfile(out_dir, sprintf('run_%s.mat', ts));
    save(mat_path, 'run_result');

    txt_path = fullfile(out_dir, sprintf('run_%s.txt', ts));
    fid = fopen(txt_path, 'w');
    fprintf(fid, 'Config   : %s\n', run_result.config_name);
    fprintf(fid, 'UUID     : %s\n', run_result.run_uuid);
    fprintf(fid, 'Mode     : %s\n', run_result.mode);
    fprintf(fid, 'Start    : %s\n', char(run_result.start_time));
    fprintf(fid, 'End      : %s\n', char(run_result.end_time));
    fprintf(fid, 'Duration : %.2f s\n', run_result.duration_s);
    fprintf(fid, 'Status   : %s\n', run_result.status);
    if ~isempty(run_result.error_message)
        fprintf(fid, 'Error    : %s\n', run_result.error_message);
    end
    fprintf(fid, '\nImages processed: %d\n', numel(run_result.image_results));
    for i = 1:numel(run_result.image_results)
        ir = run_result.image_results(i);
        fprintf(fid, '  [%d] %s  (%d seq, %.10f s)\n', i, ir.name, ir.n_sequences, ir.duration_s);
    end
    fclose(fid);

    fprintf('  Results saved: %s\n', mat_path);
end


function print_run_summary(r)
    fprintf('\n--- Run Summary: %s ---\n', r.config_name);
    fprintf('  Status   : %s\n', r.status);
    fprintf('  Duration : %.2f s\n', r.duration_s);
    fprintf('  Images   : %d\n', numel(r.image_results));
    total_seq = sum(arrayfun(@(x) x.n_sequences, r.image_results));
    fprintf('  Sequences: %d\n\n', total_seq);
end


function banner(msg, quiet)
    if quiet, return; end
    line = repmat('#', 1, 70);
    fprintf('\n%s\n# %s\n%s\n\n', line, msg, line);
end


function uuid = generate_uuid()
% Generate a simple pseudo-UUID string (no Toolbox dependency).
    rng('shuffle');
    hex_chars = '0123456789abcdef';
    n = 32;
    idx = randi(16, 1, n);
    raw = hex_chars(idx);
    uuid = sprintf('%s-%s-%s-%s-%s', raw(1:8), raw(9:12), raw(13:16), raw(17:20), raw(21:32));
end

function progress_bar(si, n_seq, elapsed, mode, quiet)
% Print an in-place progress bar with ETA to stdout.
    if quiet, return; end
    WIDTH  = 30;
    filled = round(WIDTH * si / n_seq);
    bar    = [repmat('=', 1, filled) repmat(' ', 1, WIDTH - filled)];
    rate   = si / max(elapsed, 1e-9);
    eta_s  = (n_seq - si) / max(rate, 1e-9);
    if si < n_seq
        eta_str = sprintf('ETA %ds', round(eta_s));
    else
        eta_str = 'done    ';
    end
    fprintf('\r    [%s] %d/%d  %.10f seq/s  %s  (%s)', ...
        bar, si, n_seq, rate, eta_str, mode);
    if si == n_seq, fprintf('\n'); end
end
