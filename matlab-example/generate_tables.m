function tables = generate_tables(config_name, output_dir)
% GENERATE_TABLES  Load HMM probability tables from numpy-exported .mat files.
%
% This version does NOT generate tables from scratch.  Instead it reads
% pre-converted .mat files that were produced by convert_numpy_tables.py
% from the numpy exports in ../image-sparse/models/<config>/tables/.
%
% Usage:
%   tables = generate_tables('SMALL');
%   tables = generate_tables('MEDIUM', 'my_output_dir');
%
% Inputs:
%   config_name - string: 'TINY','SMALL','MEDIUM','LARGE','XLARGE','MASSIVE'
%   output_dir  - (optional) directory containing the .mat files.
%                 Default: models/<config_lower>/tables
%
% Output:
%   tables - struct with fields:
%       initialProb    (num_positions x num_buckets x num_textures)
%       emissionProb   (256 x num_buckets)
%       spatialWeight  (num_positions x num_positions, sparse)
%       intensityWeight(num_buckets x num_buckets)
%       textureWeight  (num_textures x num_textures)
%       positionBias   (num_positions x 1)
%       bucketBias     (num_buckets x 1)
%       textureBias    (num_textures x 1)

    %% --- Validate config name ---
    valid_configs = {'TINY','SMALL','MEDIUM','LARGE','XLARGE','MASSIVE'};
    config_name   = upper(strtrim(config_name));
    if ~ismember(config_name, valid_configs)
        error('Unknown config "%s". Available: %s', ...
              config_name, strjoin(valid_configs, ', '));
    end

    %% --- Resolve table directory ---
    if nargin < 2 || isempty(output_dir)
        output_dir = fullfile('models', lower(config_name), 'tables');
    end

    if ~exist(output_dir, 'dir')
        error(['Table directory not found:\n  %s\n\n' ...
               'Run the Python converter first:\n' ...
               '  python convert_numpy_tables.py %s\n\n' ...
               'Or supply the correct path as the second argument.'], ...
               output_dir, config_name);
    end

    fprintf('=======================================================\n');
    fprintf('TRELLIS TABLE LOADER (numpy → mat)\n');
    fprintf('=======================================================\n');
    fprintf('Config    : %s\n', config_name);
    fprintf('Directory : %s\n', output_dir);
    fprintf('=======================================================\n\n');

    t_start = tic;

    %% --- Table manifest ---
    % Each entry: {field_name, file_name, is_sparse}
    manifest = {
        'initialProb',     'initialProb.mat',     false;
        'emissionProb',    'emissionProb.mat',     false;
        'spatialWeight',   'spatialWeight.mat',    true;
        'intensityWeight', 'intensityWeight.mat',  false;
        'textureWeight',   'textureWeight.mat',    false;
        'positionBias',    'positionBias.mat',     false;
        'bucketBias',      'bucketBias.mat',       false;
        'textureBias',     'textureBias.mat',      false;
    };

    total_bytes = 0;

    for i = 1:size(manifest, 1)
        field     = manifest{i, 1};
        fname     = manifest{i, 2};
        is_sparse = manifest{i, 3};

        fpath = fullfile(output_dir, fname);

        if ~exist(fpath, 'file')
            error(['Missing table file:\n  %s\n\n' ...
                   'Run convert_numpy_tables.py to generate all .mat files.'], fpath);
        end

        fprintf('[%d/%d] Loading %s ...\n', i, size(manifest,1), field);

        loaded = load(fpath, 'data');   % file always contains a variable called 'data'
        data   = loaded.data;

        % scipy.io.savemat may store sparse arrays as MATLAB sparse already;
        % if not (stored as dense), convert explicitly.
        if is_sparse && ~issparse(data)
            data = sparse(data);
        end

        tables.(field) = data;

        nb = whos('data').bytes;
        total_bytes = total_bytes + nb;

        if is_sparse
            fprintf('      Shape: [%d x %d]  Non-zeros: %d  Sparsity: %.4f%%\n\n', ...
                size(data,1), size(data,2), nnz(data), ...
                (1 - nnz(data) / numel(data)) * 100);
        else
            fprintf('      Shape: [%s]  Memory: %.2f KB\n\n', ...
                num2str(size(data)), nb/1024);
        end
    end

    %% --- Load metadata (optional) ---
    meta_path = fullfile(output_dir, 'metadata.mat');
    if exist(meta_path, 'file')
        meta = load(meta_path, 'metadata');
        tables.metadata = meta.metadata;
        fprintf('Metadata loaded from metadata.mat\n\n');
    else
        fprintf('(No metadata.mat found – skipping)\n\n');
    end

    elapsed = toc(t_start);

    %% --- Summary ---
    fprintf('=======================================================\n');
    fprintf('SUMMARY\n');
    fprintf('=======================================================\n');
    fprintf('Tables loaded    : %d\n',    size(manifest, 1));
    fprintf('Total memory     : %.2f MB\n', total_bytes/1024^2);
    fprintf('Load time        : %.2f s\n',  elapsed);
    fprintf('=======================================================\n');
    fprintf('All tables loaded successfully.\n\n');
end
