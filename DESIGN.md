---
name: ARP — Typographic Workbench
description: Readable restoration comparison with independent, combinable states.
colors:
  ground: "#141414"
  surface: "#202020"
  text: "#f4f3ef"
  secondary: "#c3c2bc"
  comparison: "#244c70"
  rendered: "#205358"
  pipeline: "#2c593d"
  active: "#c6def2"
typography:
  title:
    fontFamily: "Source Sans 3"
    fontSize: "24px"
    fontWeight: 600
    lineHeight: 1.2
  section:
    fontFamily: "Source Sans 3"
    fontSize: "18px"
    fontWeight: 600
  method:
    fontFamily: "Source Sans 3"
    fontSize: "16px"
    fontWeight: 600
  control:
    fontFamily: "Source Sans 3"
    fontSize: "15px"
    fontWeight: 500
  status:
    fontFamily: "Source Sans 3"
    fontSize: "13px"
rounded:
  method: "4px"
  control: "5px"
spacing:
  row-inline: "8px"
  workspace: "16px"
---

# Design System: ARP comparison workbench

## Overview

**Typographic Workbench.** This system describes the shared Remove Noise,
Light Deflicker and Enhancement comparison surface, not every legacy screen.
It is an Operate interface for repeated restoration decisions. Source Sans 3
and Frutiger-inspired legibility, orientation and proportion are user commitments;
the palette is not represented as a historical Frutiger palette.

## Colors

Keep video surrounds neutral. In Review, a blue method row identifies the one
inspected candidate, even without a rendered result. In Labor, blue switches
identify multi-comparison membership. Petrol identifies rendered availability,
and a green method row identifies the pipeline choice. These properties combine:
a green row retains its colour and gains an In view label when inspected in
Review; in Labor the inspection label says Inspecting. Text always names the state.
Amber is caution, light red is failure, and successful copy feedback is green.

## Typography

Use the bundled variable Source Sans 3 and its existing OFL provenance. Method
names anchor the list. Keep status text at 13px, even when nearby actions are
larger. Compact comparison switches and matrix controls have explicitly smaller
roles; do not reduce the whole interface to fit. Tabular figures belong on
changing times and measurements. Real command text may remain monospace.

## Layout

Desktop has a 320px inspector and flexible image comparison, separated by 16px.
The inspector scrolls independently of its action footer. Below 1100px it opens
as a native dialog. Below 600px the images stack. Preserve the image position,
zoom, viewport, membership and pipeline choice when changing presentation.

Keep the standard inspector compact: one Methods heading, short method rows,
three method groups with strength buttons for Remove Noise, and only relevant
parameters. More actions & details retains pipeline rendering, export and paths.
Repeated selected-result headings and empty status lines do not reserve space.
At short heights, tighten spacing rather than shrinking body text; expanded
details, warnings and very small windows retain safe scrolling.

## Elevation & Depth

Flat tonal surfaces; no ornamental glow. The native inspector dialog uses a
backdrop. Never animate, tint or crossfade film pixels for decorative feedback.

## Shapes

Methods form an open list rather than checkbox tiles. Compare in lab/In lab
comparison is an explicitly labelled multiple-selection control, visible only
in Labor. Review inspection does not alter retained Labor membership. Pipeline
is a green row plus a text tag, not an inferred colour-only state.

## Components

Method name or row inspects any method; preview availability does not disable
selection. Missing results say Not rendered and offer an explicit render request
for the inspected Review candidate (or missing Labor members). Adoption remains
an explicit footer action. Pending adoption says Adopting; green follows server
confirmation only. Keep one contextual About this method disclosure, retaining
scope and limitations. Do not decorate each method/field with an info button.
Method names and parameter labels/controls have hover/focus explanations. Each
algorithm has a small clickable Effort low/medium/high badge, green/yellow/red
with text (foreground/background: #e0f5e7/#214c35, #fff0be/#5b481b,
#ffe4de/#642d2a). This is qualitative processing cost, not quality or a timing
benchmark; CPU/CUDA, requirements and cost drivers are explained in the help.
Only the badge uses effort colour; blue inspection and green pipeline rows stay
independent. Clicking the badge opens help without selecting or rendering.
Keep contextual experimental/model warnings and required confirmations visible.
Alt+wheel guidance stays beside the image controls, outside the analysis drawer.
Native 1:1 must have a matched foreground/background pair in its pressed state.
Tab clicks immediately replace the old workspace with the target heading and
loading status. Show elapsed waiting time without invented progress percentages;
failed requests offer retry. Navigation remains available and late replies cannot
replace the newly selected tab.

## Do's and Don'ts

- Keep availability, inspection, membership and pipeline adoption independent.
- Verify combined states, disabled controls, keyboard focus and narrow layouts.
- Preserve original/stage-input references and signed processing provenance.
- Do not duplicate explanations across custom hover, info buttons and titles.
- Do not mark success messages as errors or sacrifice contrast for muted styling.
