# Global design rules

Principles that hold for every project. Short on purpose: no hex codes, no font names, no component specs. Those live in each project's `DESIGN.md`.

If a project's `DESIGN.md` conflicts with anything here, `DESIGN.md` wins.

## Direction
- Choose a direction and name it before drawing anything (for example "dark, dense, precise"). Never fall back to a default look.
- Every screen should look like it came from one hand. Consistency beats novelty.

## Type
- One typeface family per project. Build hierarchy from size, weight and tracking, not from extra fonts.
- Large display text gets tighter tracking. Small text stays readable first.
- Use a fixed type scale. No one-off sizes.

## Color
- Neutral palette plus at most one accent. No gradients unless the project asks for them.
- At most 4 text greys. Hierarchy comes from grey steps and size.
- Status colors (success, warning, error) are the only extra hues, and only where they carry meaning.

## Layout and surfaces
- Separate with spacing and subtle surface steps, not borders and boxes around everything.
- Use a fixed spacing scale and a small set of radii. Do not mix many corner styles.
- Avoid the evenly spaced grid of identical cards. Vary scale and emphasis to show what matters.

## Motion
- Motion explains a change; it never decorates.
- Feedback (hover, press, toggle): 100 to 200ms. Panels and large moves: up to about 700ms.
- Use custom ease-out curves that start fast and settle softly. Do not use default `ease` or `linear` for UI.
- Animate only transform and opacity where possible.
- Honor `prefers-reduced-motion`.

## States and content
- Design every state: empty, loading, error, long text, no permission, offline.
- Use real or realistic content in mockups. No lorem ipsum, no "Item 1".

## Accessibility
- Text contrast at least 4.5:1 (3:1 for large text).
- Visible focus on every interactive element. Full keyboard support.
- Touch targets at least 40px on touch devices.

## Anti-slop checklist
Reject the design if it has any of these without a stated reason:
- Default font at default weight and spacing
- Purple or blue gradient hero
- Glassmorphism or heavy shadows on everything
- Emoji used as icons
- Uniform rounded cards with an icon, a title and two lines of text
- Centered everything with no clear hierarchy
