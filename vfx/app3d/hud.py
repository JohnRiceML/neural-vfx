"""HUD text + budget bar (pygame). Rebuilt a few times per second from host-side numbers only."""
from __future__ import annotations

import pygame

HELP = [
    "RMB drag orbit   wheel zoom   LMB drag push smoke",
    "F  tap: fire on/off at cursor, hold: torch",
    "G  smoke puff at cursor",
    "N  cycle mode   V side-by-side   B cycle right view",
    "T  march scale  1-4 presets  O obstacle  R reset",
    "Space pause   H hud   C auto-orbit   Esc quit",
]
STAGE_COLORS = {"sim": (110, 170, 255), "upscale": (150, 220, 150), "detail": (230, 160, 255),
                "render": (255, 200, 110)}


class Hud:
    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 17)
        self.big = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 22, bold=True)
        self.small = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 15)

    def label(self, surf, text: str, pos, color=(235, 235, 240), sub: str | None = None, sub_color=(255, 120, 110)):
        t = self.big.render(text, True, color)
        s = self.font.render(sub, True, sub_color) if sub else None
        w = max(t.get_width(), s.get_width() if s else 0) + 14
        h = t.get_height() + (s.get_height() + 2 if s else 0) + 8
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
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

    def panel(self, lines: list, budget: dict | None, width: int = 560, help_on: bool = True) -> pygame.Surface:
        """lines: list of (text, color) or None for a spacer. budget: {'stages': [(name, ms)], 'limit': ms}."""
        items = list(lines)
        if help_on:
            items += [None, ("controls", (150, 160, 180))] + [(h, (170, 175, 190)) for h in HELP]
        lh = self.font.get_linesize()
        height = 16 + sum(10 if it is None else lh for it in items) + (46 if budget else 0)
        surf = pygame.Surface((width, height), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 140))
        y = 8
        for it in items:
            if it is None:
                y += 10
                continue
            text, col = it
            f = self.small if col == (170, 175, 190) else self.font
            surf.blit(f.render(text, True, col), (10, y))
            y += lh
            if budget and text.startswith("total"):
                y = self._bar(surf, budget, y + 4, width - 20) + 4
        return surf

    def _bar(self, surf, budget, y, w) -> int:
        limit = budget["limit"]
        scale = w / (limit * 1.5)                    # bar spans 1.5x the budget
        x = 10
        pygame.draw.rect(surf, (40, 42, 50), (10, y, w, 16))
        for name, ms in budget["stages"]:
            ww = int(ms * scale)
            if ww > 0:
                pygame.draw.rect(surf, STAGE_COLORS.get(name, (200, 200, 200)), (x, y, min(ww, 10 + w - x), 16))
                x += ww
                if x >= 10 + w:
                    break
        lx = 10 + int(limit * scale)
        pygame.draw.line(surf, (255, 255, 255), (lx, y - 3), (lx, y + 19), 2)
        lab = self.small.render(f"{limit:.0f} ms budget", True, (220, 220, 230))
        surf.blit(lab, (10 + w - lab.get_width(), y + 18))
        xx = 10
        for name, _ in budget["stages"]:
            t = self.small.render(name, True, STAGE_COLORS.get(name, (200, 200, 200)))
            surf.blit(t, (xx, y + 18))
            xx += t.get_width() + 10
        return y + 36
