"""HUD (pygame): a dedicated strip below the views, so it never covers a view in any mode.

Three columns: status | GPU budget (per-stage table + bars) | key help. Rebuilt a few times per
second from host-side numbers only.
"""
from __future__ import annotations

import pygame

HELP = [
    "RMB drag  orbit        wheel  zoom",
    "LMB drag  push smoke   G  smoke puff",
    "F  tap: fire on/off    hold: torch",
    "N  main mode   B  right mode   V  side-by-side",
    "1-4 presets  O obstacle  R reset  T march scale",
    "Q detail quality   J temporal march",
    "P fused pressure   K conf@128 (solver)",
    "Space pause  C auto-orbit  H hud  Esc quit",
]
STAGE_COLORS = {"sim": (110, 170, 255), "detail": (150, 220, 150), "render": (255, 200, 110)}
STRIP_BG = (14, 15, 20)
DIM, TXT, HEAD = (150, 158, 175), (225, 228, 235), (255, 210, 120)
OK, BAD = (140, 220, 140), (255, 120, 110)


class Hud:
    def __init__(self):
        pygame.font.init()
        mono = "consolas,dejavusansmono,couriernew"
        self.font = pygame.font.SysFont(mono, 16)
        self.big = pygame.font.SysFont(mono, 20, bold=True)
        self.small = pygame.font.SysFont(mono, 14)
        self.lh = self.font.get_linesize() + 1

    # ------------------------------------------------------------------ on-view labels
    def label(self, surf, text: str, pos, color=(235, 235, 240), sub: str | None = None, sub_color=(255, 120, 110)):
        t = self.big.render(text, True, color)
        s = self.small.render(sub, True, sub_color) if sub else None
        w = max(t.get_width(), s.get_width() if s else 0) + 14
        h = t.get_height() + (s.get_height() + 2 if s else 0) + 8
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 140))
        surf.blit(bg, pos)
        surf.blit(t, (pos[0] + 7, pos[1] + 4))
        if s:
            surf.blit(s, (pos[0] + 7, pos[1] + 6 + t.get_height()))

    def center_note(self, surf, rect, lines, color=(255, 130, 120)):
        x, y, w, h = rect
        yy = y + h // 2 - len(lines) * 14
        for i, ln in enumerate(lines):
            t = (self.big if i == 0 else self.font).render(ln, True, color if i == 0 else (200, 200, 210))
            surf.blit(t, (x + (w - t.get_width()) // 2, yy))
            yy += t.get_height() + 6

    # ------------------------------------------------------------------ strip
    def strip(self, size, status: list, budget: dict) -> pygame.Surface:
        """status: [(text, color) | None]; budget: see App.budget_info()."""
        W, H = size
        surf = pygame.Surface((W, H))
        surf.fill(STRIP_BG)
        pygame.draw.line(surf, (60, 64, 78), (0, 0), (W, 0), 2)
        c1, c2, c3 = 14, 440, W - 440
        self._lines(surf, status, c1, 10)
        self._budget(surf, budget, c2, 10, c3 - c2 - 24)
        pygame.draw.line(surf, (40, 43, 54), (c2 - 14, 10), (c2 - 14, H - 10), 1)
        pygame.draw.line(surf, (40, 43, 54), (c3 - 14, 10), (c3 - 14, H - 10), 1)
        self._lines(surf, [("controls", DIM)] + [(h, TXT) for h in HELP], c3, 10)
        return surf

    def _lines(self, surf, lines, x, y):
        for it in lines:
            if it is None:
                y += 6
                continue
            surf.blit(self.font.render(it[0], True, it[1]), (x, y))
            y += self.lh
        return y

    def _budget(self, surf, b: dict, x, y, w):
        """Per-stage table: ms this tick | ms per 60 Hz frame (amortized) | stage budget, with a bar per
        stage (solid = per frame, outline = this tick, white mark = stage budget) and a stacked
        per-frame total bar against the frame budget."""
        f = self.font
        cols = [x, x + 190, x + 280, x + 370]          # name | this tick | per frame | budget
        bar_x, bar_w = x + 440, w - 440
        hdr = [(b["title"], HEAD, cols[0]), ("tick", DIM, cols[1]), ("/frame", DIM, cols[2]), ("budget", DIM, cols[3])]
        for t, col, xx in hdr:
            surf.blit(f.render(t, True, col), (xx, y))
        surf.blit(self.small.render(f"0 .. {b['stage_scale']:.0f} ms", True, DIM), (bar_x, y + 2))
        y += self.lh
        sc = bar_w / b["stage_scale"]
        for name, label, tick, per, budget in b["rows"]:
            ok = per <= budget
            surf.blit(f.render(label, True, STAGE_COLORS[name]), (cols[0], y))
            surf.blit(f.render(tick, True, TXT), (cols[1], y))
            surf.blit(f.render(f"{per:5.2f}", True, OK if ok else BAD), (cols[2], y))
            surf.blit(f.render(f"{budget:4.1f}", True, DIM), (cols[3], y))
            by, bh = y + 3, self.lh - 7
            pygame.draw.rect(surf, (34, 36, 46), (bar_x, by, bar_w, bh))
            pygame.draw.rect(surf, STAGE_COLORS[name], (bar_x, by, min(bar_w, int(per * sc)), bh))
            tw = min(bar_w, int(b["tick_ms"][name] * sc))
            if tw > 0:
                pygame.draw.rect(surf, STAGE_COLORS[name], (bar_x, by, tw, bh), 1)
            bx = bar_x + int(budget * sc)
            pygame.draw.line(surf, (255, 255, 255), (bx, by - 2), (bx, by + bh + 1), 2)
            y += self.lh
        # frame total
        tot, lim, tick_tot = b["frame_ms"], b["limit"], b["tick_total"]
        ok = tot <= lim
        surf.blit(f.render("frame total", True, TXT), (cols[0], y))
        surf.blit(f.render(f"{tick_tot:5.2f}", True, TXT), (cols[1], y))
        surf.blit(f.render(f"{tot:5.2f}", True, OK if ok else BAD), (cols[2], y))
        surf.blit(f.render(f"{lim:4.1f}", True, DIM), (cols[3], y))
        y += self.lh + 2
        # stacked per-frame bar vs the 6 ms budget, stage budget allocations as ticks below
        full_w = cols[3] + 50 - x + bar_w + 20
        scale = full_w / (lim * 1.5)
        bh = 14
        pygame.draw.rect(surf, (34, 36, 46), (x, y, full_w, bh))
        xx = x
        for name, _, _, per, _ in b["rows"]:
            ww = int(per * scale)
            if ww > 0:
                pygame.draw.rect(surf, STAGE_COLORS[name], (xx, y, min(ww, x + full_w - xx), bh))
                xx += ww
                if xx >= x + full_w:
                    break
        tw = min(full_w, int(tick_tot * scale))
        pygame.draw.rect(surf, (200, 200, 210), (x, y, tw, bh), 1)
        acc = 0.0
        for name, _, _, _, budget in b["rows"]:
            acc += budget
            ax = x + int(acc * scale)
            pygame.draw.line(surf, (255, 255, 255) if abs(acc - lim) < 1e-6 else (120, 124, 140),
                             (ax, y - 3), (ax, y + bh + 3), 3 if abs(acc - lim) < 1e-6 else 1)
        y += bh + 3
        note = self.small.render(b["note"], True, DIM)
        surf.blit(note, (x, y))
        lab = self.small.render(f"{lim:.0f} ms frame budget", True, (230, 230, 235))
        surf.blit(lab, (x + int(lim * scale) - lab.get_width() // 2, y))
        y += self.small.get_linesize() + 2
        for text, col in b["extra"]:
            surf.blit(f.render(text, True, col), (x, y))
            y += self.lh
