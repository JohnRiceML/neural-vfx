// present.hlsl - the demo's only own GPU pass: copy the frame (renderer COLOR or detail output) into the back
// buffer and overlay the HUD text, in one full-screen triangle. OWNER: native app engineer.
// The HUD costs only ALU inside its rectangle: the text grid (64 x 16 cells, 1 byte each: glyph 0-63 = ASCII 32-95,
// top 2 bits = palette colour) and the 5x7 font live in the root constant buffer (1.7 KB, rewritten per frame);
// no texture, no vertex buffer, no readback.

cbuffer Hud : register(b0) {
    uint4  g_font[32];      // 64 glyphs x uint2: .xy = glyph 2i, .zw = glyph 2i+1 (rows 0-3 | rows 4-6, 8 bits/row)
    uint4  g_text[64];      // 1024 cells, row-major, 4 cells per uint
    float4 g_palette[4];    // text colours
    float4 g_backdrop;      // rgb, opacity of the box behind the text
    float4 g_clear;         // colour when there is no source (--solver-only)
    int4   g_box;           // backdrop rect x0, y0, x1, y1 (exclusive)
    int4   g_grid;          // text origin x, y, pixel scale, flags (1 = HUD on, 2 = has source)
    int4   g_dims;          // cols, rows, 0, 0
};

Texture2D<float4> g_src : register(t0);

float4 vs_main(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2(float((id << 1) & 2), float(id & 2));
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

uint pick(uint4 v, uint i) {
    return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
}

float4 ps_main(float4 pos : SV_Position) : SV_Target {
    int2 p = int2(pos.xy);
    int flags = g_grid.w;
    float3 c = (flags & 2) != 0 ? g_src.Load(int3(p, 0)).rgb : g_clear.rgb;
    if ((flags & 1) != 0 && all(p >= g_box.xy) && all(p < g_box.zw)) {
        c = lerp(c, g_backdrop.rgb, g_backdrop.a);
        int2 q = p - g_grid.xy;
        if (all(q >= 0)) {
            int s = max(g_grid.z, 1);
            int2 u = q / s;                           // font pixels; a cell is 6 x 9 of them (5 x 7 glyph + gaps)
            int2 cell = u / int2(6, 9);
            int2 sub = u - cell * int2(6, 9);
            if (cell.x < g_dims.x && cell.y < g_dims.y && sub.x < 5 && sub.y < 7) {
                uint idx = uint(cell.y * g_dims.x + cell.x);
                uint word = pick(g_text[idx >> 4], (idx >> 2) & 3);
                uint b = (word >> ((idx & 3) * 8)) & 0xFF;
                uint g = b & 63;
                if (g != 0) {
                    uint4 fp = g_font[g >> 1];
                    uint2 fw = (g & 1) != 0 ? fp.zw : fp.xy;
                    uint r = uint(sub.y);
                    uint rowbits = ((r < 4 ? fw.x : fw.y) >> ((r & 3) * 8)) & 0xFF;
                    if (((rowbits >> (4 - uint(sub.x))) & 1) != 0)
                        c = g_palette[b >> 6].rgb;
                }
            }
        }
    }
    return float4(c, 1.0);
}
