---
target: Remove Noise tab
total_score: 19
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 3
target_identity: "file:I:\\ai-remaster-pipeline\\ai_remaster_gui\\static\\js\\render-denoise.js"
target_fingerprint: "sha256:75febfa0ab24cba77f3f70afe91217e67d76619008029ad8323062a3a10cfd3d"
target_path: "I:\\ai-remaster-pipeline\\ai_remaster_gui\\static\\js\\render-denoise.js"
timestamp: 2026-11-03T12-08-42Z
slug: ai-remaster-gui-static-js-render-denoise-js
---
Method: dual-agent (A: ses_e5eb83945fferEzbKQ0EquvkXY · B: ses_e5eb83944ffelG4CSJ6h3oCWmi)

## Design Health Score

| # | Heuristik | Score | Kernproblem |
|---|---|---:|---|
| 1 | Visibility of System Status | 2/4 | `Ready 100%`, `9/9 Varianten fertig` und zwei aktive Run-CTAs gleichzeitig; „Load same section“ hat keinen Pending-State. |
| 2 | Match System / Real World | 2/4 | Domänensprache ist stark, aber Varianten sind Hashed-Pfade und die Oberfläche spricht zwei Sprachen. |
| 3 | User Control and Freedom | 3/4 | Stop, Rerun, Zoom, Wipe und Lightbox existieren; keine „Variante übernehmen“-Aktion, Frame geht beim Wechsel verloren. |
| 4 | Consistency and Standards | 1/4 | Englisch/Deutsch im selben Toolbar, drei Namen für die Stage, zwei Diff-Farbschemata, zwei Stops. |
| 5 | Error Prevention | 2/4 | Gute Partial-/Dimensions-Checks; `Regenerate` + `--force` und der 9/9-Batch zeigen keine Kosten-/Reuse-Vorschau. |
| 6 | Recognition Rather Than Recall | 1/4 | Neun Varianten existieren nur als Strings in vier 9–10-Optionen-Dropdowns; keine Matrix, keine Thumbnails. |
| 7 | Flexibility and Efficiency | 2/4 | ZIP und Signatur-Reuse helfen; keine Shortcuts, kein Loop, kein Frame-Lock, kein Batch-Vergleich. |
| 8 | Aesthetic and Minimalist Design | 2/4 | Kohärente dunkle Palette, aber flacher Kartenstapel, ~350 px tote Spalte, 10–12-px-Belegtext, Spenden-Button am lautesten. |
| 9 | Error Recovery | 2/4 | Konkrete, ehrliche Meldungen; sie erscheinen nur transient in einer Statuszeile ohne nächsten Schritt. |
| 10 | Help and Documentation | 2/4 | Reichlich Inline-Caveats, aber als Mikrotexte verstreut; kein Workflow-Guide, kein Doku-Link. |
| **Total** | | **19/40** | **Poor** |

## Design Specificity Verdict

**LLM assessment:** Der Inhalt ist unverkennbar ARP – exakt dekodierte Frames statt H.264-Preview, „ORIGINAL · Render-Eingang“, 0,18/255-Luminanzdeltas, „Kornbeurteilung separat vornehmen“, „Keine Defektmaske“, Signatur-Reuse und ein ZIP mit Übersichtsgrid, Frame-Grids, Diff-PNGs, CSV und Legende. Das könnte kein generisches Videoschnitt-Tool schreiben.

Die Hülle ist dagegen austauschbar: links ein Einstellungsformular, rechts Karten, darunter ein Vollbreiten-Player, als Identität dienen rohe Dateipfade. Das eigentliche Mentalmodell – 3 Methoden × 3 Stärken – ist ein Absatz, ein Button und ein Zähler. Es gibt keine Matrix, keine Thumbnails, kein sichtbares Set von neun Varianten. Dazu kommt eine sichtbare Naht: linke Karte und A/B/C sind englisch, Batch und Exact Review deutsch. „Ready 100%“ steht neben „9/9 Varianten fertig“, „Ansicht“ neben „Display zoom“ in derselben Leiste.

**Deterministic scan:** Der statische Scan über `index.html` endet mit Exit 0 und einem Advisory: `repeating-stripes-gradient` aus `styles.css:25`, das nur auf dem Overview-Tab gerendert wird – für diesen Tab außerhalb des Scopes. Der gerenderte Scan über CDP findet **13 Findings** (6 Primary, 7 Advisory): 2× `low-contrast` (Primary-Buttons, 3,54:1), 2× `line-length` (91–100 Zeichen), 1× `tiny-text` (11 px), 1× `nested-cards`, 5× `tiny-text` advisory (10–11,5 px), 1× Platzhalter-Kontrast, 1× `straggler`. Die Live-Overlay-Injektion war erfolgreich; der Console-Report enthält dieselben 13 Einträge. Es bleibt kein sichtbares Overlay zurück (Headless-Tab geschlossen), daher kein irreführender „[Human]-Tab“-Hinweis.

**Abgleich:** Beide Assessments treffen sich beim Primärbutton-Kontrast (3,54:1, WCAG AA verfehlt), bei zu kleinen Belegtexten und beim flachen Hierarchie-Eindruck. Der Detektor fand zusätzlich die dreifach verschachtelten Karten, die 91–100-Zeichen-Zeilen und den Straggler – Dinge, die im Design-Review untergingen. Das Design-Review fand dafür das eigentliche Problem, das kein Regelwerk sieht: Der Chart normalisiert 0,36/255 zu einer Wand aus Vollhöhen-Balken, und die Seite hat keine Commit-Aktion. Falsch-positive: `broken-image` ×6 und `flat-type-hierarchy` aus dem Fragment-Scan (Laufzeit-`src` bzw. fehlendes Stylesheet), `repeating-stripes` außerhalb des Tabs. Vier Findings haben nur zwei Ursachen (Primary-Button-Regel, `.field-help`).

## Overall Impression

Ein starkes Spezialwerkzeug in einem generischen Dashboard. Der Exact-Frame-Inspector ist der Burggraben: echte dekodierte Frames, 1-basierte Frame-Anzeige, ±1/±10, Original vs. Render-Eingang, Diff in Quellauflösung. Aber die Seite öffnet auf einem Einstellungsformular und einer Textkarte, nicht auf dem Vergleich. Der größte Hebel: die neun Varianten zu einer sichtbaren 3×3-Entscheidungsfläche machen, den Vergleich nach oben holen und dem Ganzen eine „Übernehmen“-Aktion geben.

## What's Working

1. **Der Exact-Frame-Inspector ist der Differenzierer.** Dekodierte Frames, Frame-Readout, ±1/±10, „ORIGINAL · Render-Eingang“ statt bloß „Original“, Diff in Quellauflösung und ein Wipe-Grip, der an den Rändern ausblendet. Das kommt nur von Leuten, die das Werkzeug selbst benutzen.
2. **Die Ehrlichkeitsschicht ist on-brand und selten.** „Gemessene Pixeländerungen sind kein Beweis für Rausch- oder Kornentfernung“, „Bei Langfilmen nur Stichproben“, „Browser-Synchronisation ist annähernd“. Das verhindert die gefährlichste Fehlinterpretation genau in dem Moment, in dem ein verstärktes Bild betrachtet wird.
3. **Batch + ZIP antizipieren den echten Workflow.** Neun Varianten, Signatur-Reuse, Original bleibt unverändert, Export mit Übersichtsgrid, Frame-Grids, Diff-PNGs, CSV und Legende. Das macht aus einem visuellen Vergleich einen belegbaren Nachweis.

## Priority Issues

### [P0] Keine Commit-Aktion – die Review kann nicht abgeschlossen werden
**What:** Jede Kontrolle ändert nur, was man *sieht*. Nichts ändert, was die Stage *schreibt*. Im gerenderten Zustand vergleicht die Exact Review `bm3d_balanced`, während links `Output Path` auf `bm3d_strong` steht. Die einzige Handlung ist: nach oben scrollen, zwei Dropdowns umstellen, `Regenerate` angehakt lassen und `Run Remove Noise` drücken.
**Why it matters:** Die Stage heißt „Remove Noise“, ihre Aufgabe ist die Methodenwahl. Die Fläche, die diese Wahl ermöglicht, kann sie nicht abschließen. Das erzwingt Merken, lädt zum falschen Output ein und macht den emotionalen Peak zur Sackgasse.
**Fix:** Eine persistente `Gewählte Variante`-Leiste unter der Exact Review: `BM3D · Balanced` · `Ø 0,36/255` · `28,25 % Pixel` · `[Als Stage-Ausgabe übernehmen]` · `[Einstellungen kopieren]`. „Übernehmen“ setzt die linken Felder, wählt den signaturgleichen Output und rendert nur, wenn er fehlt. Den Batch als 3×3-Radiomatrix bauen, damit gewählte Zelle und Entscheidungsleiste dasselbe Objekt sind.
**Suggested command:** `shape`

### [P1] Zwei Sprachen, drei Namen, Hashes als Identität
**What:** `<html lang="en">` umschließt deutsche Inhalte. Links und A/B/C sind englisch, Batch und Exact Review deutsch; `Ansicht` und `Display zoom` stehen in einer Leiste. Die Stage heißt `Remove Noise`, `denoise` und `Rauschreduzierung`. Varianten erscheinen als `Neighbors_denoised_bm3d_balanced_e4547406d4b6.mkv` in vier Dropdowns; Metriken nutzen `0.36` und `28.25 %` in einer deutschen Oberfläche.
**Why it matters:** WCAG 3.1.1 ist verletzt, Screenreader sprechen deutsche Sätze englisch aus. Für Sehende wirkt die Seite wie zwei Autoren. Der Hash ist der am wenigsten merkhafte Teil des Pfads und das am häufigsten wiederholte Label.
**Fix:** Deutsch als Tab-Sprache, `lang="de"`, englische Methodennamen in `<span lang="en">`. Strings zentralisieren. Jede Variante als `BM3D · Balanced` labeln; Pfad und Hash in `title` plus Copy-Button. Metriken mit `Intl.NumberFormat('de-DE')`. `Kontrast` in `Verstärkung` umbenennen, weil es die Differenz verstärkt.
**Suggested command:** `clarify`

### [P1] Der Frame-Change-Chart übertreibt die Änderung
**What:** 72 Balken (bis 240), je eine teal- und eine orange-Hälfte aus derselben Basis, Höhen auf den Peak normalisiert (`100 * x / peak`), keine Y-Achse, keine absolute Skala, keine Markierung für Stichproben. Im gerenderten Zustand ist die mittlere ΔY **0,36/255** – etwa 1,4 % eines 8-Bit-Schritts – trotzdem ist jeder Balken hoch.
**Why it matters:** Ein Fast-No-Op sieht dramatisch aus. Ein Restaurator könnte damit argumentieren, die Entrauschung habe mehr getan, als sie tat – ein direkter Widerspruch zu „keine Behauptungen, die die Pipeline nicht belegen kann“.
**Fix:** Divergierender Chart mit echter Nulllinie – aufgehellt oben, abgedunkelt unten – und beschrifteter absoluter Skala in ΔY/255, oder feste 0–2-Skala mit `Peak 0,18`-Annotation. `Peak 0,18 · Mean 0,36` über den Chart. Stichproben vs. alle Frames markieren. Ein Balken pro Frame statt zwei Hälften; Selected-Frame-Marker klar vom Keyboard-Fokus trennen; Hover-Readout.
**Suggested command:** `distill`

### [P1] Der Vergleich liegt unter dem Fold, das beste Werkzeug ist zugeklappt
**What:** `.denoise-editor-page > .denoise-compare-card` wartet auf die 918 px hohe linke Karte. Die Batch-Karte endet bei ~604 px, die A/B/C-Karte beginnt bei ~950 px; dazwischen liegen ~350 px leere rechte Spalte. Die Exact Review steckt in der A/B/C-Karte, der Wipe ist ein geschlossenes `<details>` am Ende. Der erste 1000-px-Viewport enthält null Vergleichsbilder.
**Why it matters:** Ein Vergleichswerkzeug öffnet auf einem Einstellungsformular und einer Textkarte. Der Nutzer scrollt an einer leeren Spalte zum Payoff. Die approximative H.264-Preview bekommt den besten Platz, die exakten dekodierten Frames – die der Text selbst „echte“ nennt – liegen 2.300 px tiefer. Das ist eine Umkehrung der Evidenz-Hierarchie des Produkts.
**Fix:** A/B/C in die rechte Spalte, Zeile 1/2, direkt neben die Sidebar; Batch darunter oder als 3×3-Matrix über den Playern. Die Exact Review als eigene Karte statt verschachtelt. Wipe standardmäßig offen oder als Segmented Control `Standbilder / Wischen / Differenz` oben. Eine Dreischritt-Leiste: `Varianten erzeugen → Vergleichen → Übernehmen`.
**Suggested command:** `layout`

### [P2] Fragiler State und deaktivierte Controls, die lebendig aussehen
**What:** Ein Wechsel von B ruft `denoiseReviewChoose`, setzt `denoiseReviewFrame = 0` und springt im Inspector auf Frame 1; ein Wechsel im Inspector aktualisiert B nicht. „Load same section“ erzeugt alle drei Videos neu, ohne Pending-State, und verliert die Position. Es gibt keine `button:disabled`-Regel: der Sidebar-`Stop` ist `disabled`, sieht aber wie ein aktiver amber Button aus; der Header-`Stop` ist live und identisch. `Ready 100%` (voller Progressbar) koexistiert mit `9/9 Varianten fertig` und zwei aktiven Run-CTAs; der Batch-Button sagt bei 9/9 weiter „Alle 9 Varianten rendern“.
**Why it matters:** Nutzer verlieren den Frame, den sie gerade prüfen, und können nicht erkennen, was live, pending oder fertig ist. In einer mehrstündigen Review ist ein verlorener Frame echter Aufwand; ein toter Stop ist ein Vertrauensbruch.
**Fix:** Frame beim Variantenwechsel erhalten (`denoiseReviewSelectFrame(denoiseReviewFrame, false)`), `Frame sperren`-Toggle, Inspector-Variante mit B koppeln oder explizit entkoppeln. `button:disabled { opacity: .45; cursor: not-allowed; }`, doppelten Stop entfernen oder Scope labeln. Idle-Progressbar durch `Bereit` ersetzen. Bei 9/9 CTA auf `Erneut rendern` umstellen und `8 von 9 werden wiederverwendet` zeigen.
**Suggested command:** `harden`

## Persona Red Flags

**Alex (Power User):** Um denselben Defekt in neun Varianten zu vergleichen, muss er den Frame neunmal neu finden – jeder Variantenwechsel springt auf Frame 1. Keine Matrix, keine Thumbnails, kein Pinning, keine History, kein „nächste Variante“; vier 10-Optionen-Selects und Hashes sind die einzige Navigation. Keine Shortcuts: kein Space zum Abspielen, keine Pfeiltasten, kein A/B-Toggle, keine Zahlentasten für Varianten; die 72 Chart-Balken sind 72 Tab-Stops. Kein „Einstellungen kopieren“ von der Comparison in die linke Karte.

**Sam (Accessibility/Keyboard):** `lang="en"` auf deutschen Inhalten; Screenreader sprechen die Seite falsch aus. Primärbutton-Kontrast 3,54:1 (Weiß `#edf4f6` auf `#2d8f7d`) verfehlt AA für 13,33-px-Text; Captions sind 10 px. Der 72-Balken-Chart nutzt 72 Tab-Stops, `aria-pressed` für eine Einfachauswahl und keine Roving-Tabindex-/Pfeilnavigation. „Frame-Änderungen“ und „Frame Inspector“ sind `<strong>`, keine Headings. Tabs sind einfache Buttons ohne `aria-current`, `role=tab` oder Tablist-Pattern. Mehrere `role=status`-Regionen feuern bei jedem Frame-Wechsel – ein Announcement-Flood. Der deaktivierte Stop sieht aktiv aus; das A–B-Diff-Canvas hat ein `aria-label`, aber kein textuelles Äquivalent.

**Marta (Archive Restorer, accountable to a commissioning archive):** Das A/B/C-„Original“ ist `evaluation/Neighbors_72.mkv`, während das Stage-Kommando `intermediate/source_sections/Neighbors_72_...high.mkv` liest; die Seite sagt nicht, ob das dieselben Pixel sind. Das ist der Unterschied zwischen gültigem A/B und dem Vergleich zweier Encodes. Keine Provenienz neben dem Beleg: Methode, Preset, Frame, Mean ΔY und Changed-% stehen nicht zusammen. `Regenerate` + `--force` kann einen validierten Output überschreiben; die einzige Absicherung ist ein generischer Browser-Confirm mit Pfaden. Der Readout ist clip-relativ („Frame 1/72“), nicht Source-Timecode oder Shot-Nummer. Das ZIP ist erst nach der Erzeugung sichtbar und nicht an den aktuellen Vergleich gebunden.

## Minor Observations

- Primärbutton-Kontrast **3,54:1** – `#edf4f6` auf `#2d8f7d`; dunkleres Fill (~`#1f6f61`) oder Teal nur als Border.
- **Kein Disabled-Styling:** Sidebar-Stop ist deaktiviert, aber amber und aktiv aussehend; Header-Stop ist live und identisch. Gleiches bei nicht verfügbaren ±1-Buttons und Export bei 0 Count.
- **Tabbar ist kein Tab-Pattern:** einfache Buttons, kein `aria-current`, keine Pfeilnavigation, kein gelabeltes `nav`.
- **72-Balken-Chart als Keyboard-Hindernis:** Roving Tabindex und Pfeiltasten oder ein Slider mit Frame-Readout; `aria-pressed` ist die falsche Semantik für eine Einzelauswahl.
- **Dritte Farbschsprache:** Das A–B-Preview-Diff nutzt eine Rot/Magenta-Rampe, die Exact Review Teal/Orange. Bei inkompatiblen Clips zeigt die Checkbox eine leere 512-px-Schwarzfläche ohne Erklärung.
- **AB/AC-Layouts validieren trotzdem alle drei Videos:** Ein verstecktes inkompatibles C blockiert die Wiedergabe mit „All three clips must load“; das versteckte Video dekodiert weiter. Das Diff vergleicht außerdem immer A–B, auch im A/C-Modus.
- **Performance:** Das A–B-Diff berechnet bei jedem `timeupdate` zwei 512-px-`getImageData`-Pässe. Auf Pause/Seek drosseln oder Shader nutzen, sonst ruckelt die Synchronisation, die es helfen soll.
- **Semantik:** „Frame Inspector“ ist ein `<strong>`, kein Heading; das Section-`aria-label` heißt „Drei Einzelbildansichten“ – anders als die sichtbare Überschrift. Die Seite hat keine `h1`, drei `h2` konkurrieren.
- **Doppelte Wipe-Readouts:** „50 % ORIGINAL · 50 % ERGEBNIS“ und „50 % Original links / 50 % Ergebnis rechts“. Auf eines reduzieren; die 10-px-Caption lesbar machen.
- **Command-Preview immer offen** mit `word-break: break-all`; in `<details>` mit Copy-Button – es ist Pipeline-Beweis, nicht die Hauptaufgabe.
- **Labels:** „CUDA 8.9“ ist Compute Capability 8.9, nicht die CUDA-Version → „SM 8.9“ oder „Compute 8.9“. „Start second“ → „Startzeit (s)“. „Load same section“ → „Auf alle drei anwenden“.
- **Spenden-Button** ist das hellste Element im Header, ein weißer Pill neben dem Primär-Workflow; „Quit“ steht gleichgewichtig neben „Stop“. In Settings/About verschieben bzw. destruktive Aktion trennen.
- **`Ready 100%`** ist ein unbenannter `<progress>` und im Idle voll. Im Idle entfernen oder zu „Konfiguration bereit“ machen.
- **`alert()`** für ZIP-Validierung und Batch-Fehler; inline am Feld, konsistent zum Rest.
- **„Maximal 5“ Frames** ist unerklärt und das Freitext-Kommaseparated-Feld fehleranfällig. Chips oder Timecode-Frame-Picker, und anzeigen, welche Frames im ZIP landen.
- **Selected-Frame-Outline** im Chart sieht am linken Rand wie ein Scrollbar-Thumb aus. Caret oder gefüllter Marker plus „Frame 1/72“-Label.

## Questions to Consider

1. Wenn die Karte „alle Methoden vergleichen“ heißt, warum enthält sie keine Bilder und keine Möglichkeit, einen Gewinner zu wählen?
2. Was soll der Nutzer tun, nachdem er BM3D Balanced als beste Variante erkannt hat? Wenn die Antwort „nach oben scrollen und zwei Dropdowns umstellen“ lautet – warum heißt das dann Review-Fläche?
3. Wenn 0,36/255 eine Wand aus Vollhöhen-Balken erzeugt, was kommuniziert der Chart dann außer „irgendwas ist passiert“? Würdest du diesen Chart einem Archivar zeigen, der das Ergebnis verteidigen muss?
4. Warum besitzt die approximative H.264-Preview den ersten Screen, während die exakten dekodierten Frames 2.300 px tiefer und hinter einem zugeklappten Fold liegen?
5. Ist „Ready 100%“ ein Readiness-Check, ein Fortschrittsbalken oder ein Abschlusszustand? Wenn ein Nutzer das nicht unterscheiden kann – was lügt sonst noch versehentlich?
6. Welche Sprache spricht das Produkt – und wenn die Antwort „beide“ ist: für welchen Nutzer wird hier gestaltet?
7. Warum bekommt eine optionale Stage einen Top-Level-Tab, während vier Kern-Stages keinen haben? Ist die Navigation nach Pipeline organisiert – oder danach, welche Seite jemand gebaut hat?
8. Wie oft soll ein Restaurator denselben Frame neu finden, wenn er neun Varianten vergleicht? Was kostet „jedes Mal“ über einen 90-Minuten-Film?
