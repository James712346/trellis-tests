function configs = get_configs()
% GET_CONFIGS  Returns all preset HMM configuration structs.
%
% Usage:
%   configs = get_configs();
%   cfg = configs.SMALL;
%
% Fields per config:
%   grid_width, grid_height  - image grid dimensions
%   num_buckets              - number of intensity buckets
%   num_textures             - number of texture types
%   num_cases                - number of transition cases
%   max_spatial_dist         - max allowed spatial jump
%   max_intensity_diff       - max allowed intensity bucket jump
%   max_texture_diff         - max allowed texture type jump
%   description              - human-readable label

configs.TINY.grid_width       = 16;
configs.TINY.grid_height      = 9;
configs.TINY.num_buckets      = 4;
configs.TINY.num_textures     = 2;
configs.TINY.num_cases        = 7;
configs.TINY.max_spatial_dist    = 1;
configs.TINY.max_intensity_diff  = 1;
configs.TINY.max_texture_diff    = 1;
configs.TINY.description      = 'Quick test configuration';

configs.SMALL.grid_width      = 32;
configs.SMALL.grid_height     = 18;
configs.SMALL.num_buckets     = 8;
configs.SMALL.num_textures    = 4;
configs.SMALL.num_cases       = 9;
configs.SMALL.max_spatial_dist    = 1;
configs.SMALL.max_intensity_diff  = 1;
configs.SMALL.max_texture_diff    = 1;
configs.SMALL.description     = 'Reasonable for initial testing - ultra sparse';

configs.MEDIUM.grid_width     = 48;
configs.MEDIUM.grid_height    = 27;
configs.MEDIUM.num_buckets    = 8;
configs.MEDIUM.num_textures   = 4;
configs.MEDIUM.num_cases      = 9;
configs.MEDIUM.max_spatial_dist   = 1;
configs.MEDIUM.max_intensity_diff = 1;
configs.MEDIUM.max_texture_diff   = 1;
configs.MEDIUM.description    = 'Medium complexity, ultra sparse (95%+)';

configs.LARGE.grid_width      = 64;
configs.LARGE.grid_height     = 36;
configs.LARGE.num_buckets     = 8;
configs.LARGE.num_textures    = 4;
configs.LARGE.num_cases       = 13;
configs.LARGE.max_spatial_dist    = 1;
configs.LARGE.max_intensity_diff  = 1;
configs.LARGE.max_texture_diff    = 1;
configs.LARGE.description     = 'Large state space, ultra sparse';

configs.XLARGE.grid_width     = 96;
configs.XLARGE.grid_height    = 54;
configs.XLARGE.num_buckets    = 16;
configs.XLARGE.num_textures   = 8;
configs.XLARGE.num_cases      = 13;
configs.XLARGE.max_spatial_dist   = 1;
configs.XLARGE.max_intensity_diff = 1;
configs.XLARGE.max_texture_diff   = 1;
configs.XLARGE.description    = 'Extra large, ultra sparse (>99%)';

configs.MASSIVE.grid_width    = 128;
configs.MASSIVE.grid_height   = 72;
configs.MASSIVE.num_buckets   = 16;
configs.MASSIVE.num_textures  = 8;
configs.MASSIVE.num_cases     = 17;
configs.MASSIVE.max_spatial_dist   = 9999;
configs.MASSIVE.max_intensity_diff = 9999;
configs.MASSIVE.max_texture_diff   = 9999;
configs.MASSIVE.description   = 'Massive state space - maximum sparsity!';

end
