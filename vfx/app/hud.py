"""HUD text (pygame). Rebuilt a few times per second; stats come from host-side numbers only."""
from __future__ import annotations

import pygame

HELP = [
    "N  cycle mode     V  side-by-side",
    "LMB drag  push    RMB  paint smoke",
    "F  tap: fire on/off at cursor, hold: torch",
    "1-4 presets   O obstacle   R reset",
    "Space pause   H hud   Esc quit",
]


class Hud:
    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 17)
        self.big = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 22, bold=True)
        self.small = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 15)

    def label(self, surf, text: str, pos, color=(235, 235, 240)) -> None:
        t = self.big.render(text, True, color)
        bg = pygame.Surface((t.get_width() + 12, t.get_height() + 6), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 140))
        surf.blit(bg, (pos[0], pos[1]))
        surf.blit(t, (pos[0] + 6, pos[1] + 3))

    def render_panel(self, lines: list, size, help_on: bool = True) -> pygame.Surface:
        """lines: list of (text, color) or None for a spacer."""
        surf = pygame.Surface(size, pygame.SRCALPHA)
        y = 0
        for item in lines + ([None, ("controls", (150, 160, 180))] + [(h, (170, 175, 190)) for h in HELP] if help_on else []):
            if item is None:
                y += 10
                continue
            text, col = item
            f = self.small if col == (170, 175, 190) else self.font
            surf.blit(f.render(text, True, col), (0, y))
            y += f.get_linesize()
        return surf
