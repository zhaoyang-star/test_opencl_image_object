/*
 * Copyright 2016-2019. UISEE TECHNOLOGIES (BEIJING) LTD. All rights reserved.
 * See LICENSE AGREEMENT file in the project root for full license information.
 */

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void image_object_kernel(
            __private const int N, __private const int H,
            __private const int W, __private const int C,
            __read_only image2d_t input,
            __write_only image2d_t output)
{
    const int out_channel_block_idx = get_global_id(0);
    const int out_width_idx         = get_global_id(1);
    const int out_batch_height_idx  = get_global_id(2);
    const int out_channel_blocks    = get_global_size(0);

    const int out_height_idx = out_batch_height_idx % H;

    const int pos_x = out_width_idx * out_channel_blocks + out_channel_block_idx;
    //const int pos_x = out_channel_block_idx * W + out_width_idx;
    const int pos_y = out_batch_height_idx;

    float4 in = read_imagef(input, SAMPLER, (int2)(pos_x, pos_y));

    float4 out = in + 1.f;
    write_imagef(output, (int2)(pos_x, pos_y), out);
}
