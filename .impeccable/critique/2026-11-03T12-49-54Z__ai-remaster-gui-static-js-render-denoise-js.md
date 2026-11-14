---
target: Remove Noise tab
total_score: 23
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 3
target_identity: "file:I:\\ai-remaster-pipeline\\ai_remaster_gui\\static\\js\\render-denoise.js"
target_fingerprint: "sha256:784106b65c9f1784b53cf9c38a0e7a0e5765dd79182fc2cd64426a9e81e21885"
target_path: "I:\\ai-remaster-pipeline\\ai_remaster_gui\\static\\js\\render-denoise.js"
timestamp: 2026-11-03T12-49-54Z
slug: ai-remaster-gui-static-js-render-denoise-js
---
Method: dual-agent (A: ses_e5e95a968ffe7anJLK3c5fEToY · B: ses_e5e95a967ffeo0tthKxD0aMKqt)

## Design Health Score

| # | Heuristik | Score | Kernproblem |
|---|---|---:|---|
| 1 | Visibility of System Status | 3/4 | Überall Status – Counts, Ready-Zellen, Stage-Output, Frame/fps – aber die Sidebar-Metrik folgt still dem aktuellen Frame, und der committete Stage-Output hat kein Badge auf der Matrix. |
| 2 | Match System / Real World | 3/4 | Methoden- und Preset-Vokabular passt; es fehlt aber jede Erklärung, was die Methoden mit Korn und Detail tun. „Start time (s)“ ist nicht die Sprache eines Restaurators. |
| 3 | User Control and Freedom | 3/4 | Off, A/B/C, Frame-Steps, Zoom und Wipe sind großzügig; aber die Wahl von C verschiebt still das Commit-Ziel, und es gibt kein Pin/Undo. |
| 4 | Consistency and Standards | 2/4 | 1x vs. 100%, Ready vs. rendered, Matrix-Reihenfolge vs. Dropdown-Reihenfolge, Off außerhalb der Radiogroup, zwei Diff-Farbsysteme. |
| 5 | Error Prevention | 1/4 | Die UI kann alte Pixel als neue Auswahl labeln; doppelte A/B-Slots erzeugen einen sinnlosen Null-Diff; „Render all 9“ bleibt bei 9/9 primär. |
| 6 | Recognition Rather Than Recall | 2/4 | Neun identische „Ready“-Zellen tragen keine Vergleichsdaten; B, C, Stage-Output und vorherige Variante müssen erinnert werden. |
| 7 | Flexibility and Efficiency | 3/4 | Roving-Matrix, ±1/±10, Timeline, Wipe und Batch-Export sind echte Power-Features; kein Compare-all/Pin, Chart ohne Pfeiltasten. |
| 8 | Aesthetic and Minimalist Design | 2/4 | Kohärentes dunkles Instrument, aber der 900-px-Wipe und die verstärkte Rauschkarte dominieren das Ergebnis; Wiederholung treibt die Seite auf 2.741 px. |
| 9 | Error Recovery | 2/4 | Overwrite-Confirm und Statuszeilen sind gut; fehlende/fehlgeschlagene Zustände können alte Evidenz stehen lassen, einige Fehlertexte sind noch Deutsch. |
| 10 | Help and Documentation | 2/4 | Ehrliche Caveats, aber keine Methodenführung, keine Gamma-Erklärung, und die beste Vergleichshilfe – das Batch-Grid – ist zugeklappt. |
| **Total** | | **23/40** | **Acceptable – aber mit einem vertrauenskritischen P0.** |

Vorher: **19/40**. Die vier Punkte Gewinn kommen aus Sprache, Matrix, Layout und Disabled-States – nicht aus einem besseren Entscheidungsmodell.

## Design Specificity Verdict

**LLM assessment:** Die exakte Review ist unverkennbar ARP: dekodierte Frames aus dem tatsächlichen Verarbeitungseingang, „render input“ statt „Original“, Teal/Orange-Luminanzdiff, beschrifteter ORIGINAL/RESULT-Wipe. Das kann kein generischer Video-Editor ohne Provenienz-Umbau kopieren.

Die 3×3-Matrix ist das Gegenteil: neun identische „Ready“-Zellen könnten zu jedem Plugin gehören. Sie kennt Methode × Stärke, aber nicht das, was den Restaurator interessiert – Kornverhalten, Detailerhalt, Gesichter, Geschwindigkeit. Ein Wettbewerber kopiert die Matrix in einem Nachmittag; die exakte Review würde einen Neubau erfordern. Die Identität lebt in der Review, der erste Bildschirm zeigt aber den generischen Teil.

**Deterministic scan:** Der gerenderte Scan über CDP findet **0 Findings**; der Overlay-Lauf bestätigt `[impeccable] No anti-patterns found.` Der statische Scan über `index.html` findet **1 Advisory** (`repeating-stripes-gradient`) aus totem `.workflow-cleanup`-CSS – außerhalb des Tabs. Ein Kontrolllauf auf Overview findet 12 Findings und beweist, dass der Detector nicht stillschweigend überspringt. Zusätzliche Messungen: **0 AA-Fehler**, niedrigster Kontrast **5,36:1**, **0 verschachtelte Karten**, kein horizontaler Overflow. Die 72 Chart-Balken sind **1 Tab-Stop** (71 mit `tabindex="-1"`), nicht 72 – der Tab-Observation von B war an dieser Stelle veraltet; es fehlt aber weiterhin ein Arrow-Key-Handler.

**Visual overlays:** Die Injektion war erfolgreich, der Console-Report ist sauber; im Headless-Tab bleibt kein sichtbares Overlay zurück.

## Overall Impression

Der Tab ist jetzt ehrlich und exakt – aber noch nicht entscheidbar. Die größte Baustelle ist nicht mehr die Sprache oder die Matrix, sondern die **Integrität der Evidenz**: Die Review kann Pixel der alten Variante unter dem Label der neuen zeigen. Das ist in einem Werkzeug, dessen Kernversprechen Reviewbarkeit ist, ein P0. Danach folgen die falsche 100%-Skala und die versteckte primäre Aktion.

## What's Working

1. **Exakte Provenienz ist echt.** Die Stills kommen aus dem tatsächlichen Processing-Input und dem signierten Output, nicht aus der H.264-Preview; Frame/fps/Dimensionen stehen daneben, und die Frame-Position überlebt den Variantenwechsel (verifiziert: 11/72 bleibt 11/72).
2. **Der Wipe ist das beste Instrument.** Beschriftete ORIGINAL/RESULT-Griffe, exakter Frame, ziehbares Bild plus Slider. Das ist handwerklich und für Kornbeurteilung wirklich nützlich.
3. **Die Zustandssprache ist candid.** „Render this variant“, „Off · keep the source unchanged“, „Only finished variants are included“, „Not a defect mask“, „Judge grain separately“. Der fehlende Zustand ist gestaltet, kein totes Disabled-Feld.

## Priority Issues

### [P0] Die Exact Review kann die falschen Pixel labeln
**What:** Wird **Off** gewählt, wechselt die Review-Überschrift auf „Selected: Off“ und das Panel auf „Remove Noise is off“, während ORIGINAL/RESULT/DIFF und der Readout (`Mean absolute luminance difference: 0.34 / 255 · 27.03%`) von der vorherigen Variante stehen bleiben. Das Ergebnisbild sagt weiter „RESULT · selected variant“. Derselbe Ein-Zustand-Pfad existiert für eine nicht gerenderte Zelle. Zusätzlich: Wird C auf BM3D Strong gesetzt, wandern Matrix-Ring und Review auf C, B bleibt HQDN3D, die Stage-Zeile bleibt HQDN3D – und „Use as stage output“ ist für C aktiv.
**Why it matters:** Das ist keine kosmetische Abweichung, sondern falsche Evidenz in einem Tool, dessen Kernversprechen Reviewbarkeit ist. Ein Restaurator kann eine Methode auf Basis von Pixeln einer anderen Methode freigeben.
**Fix:** Niemals das `Selected:`-Label ohne neue Pixel und neuen Readout ändern. Bei Off oder nicht gerendertem Ziel einen expliziten Empty-State einsetzen („No result for this selection; the stills below are from X“). Das Review-Ziel als ein einziges explizites Objekt führen, B/C als Referenzen labeln, die Stage-Output-Zelle badgen und C nicht still zum Commit-Kandidaten machen.
**Suggested command:** `harden`

### [P1] Die Sidebar-Metrik ist der aktuelle Frame, nicht die Variante
**What:** Unter „HQDN3D · Balanced“ steht `Mean |ΔY| 0.34 / 255 · 27.03% pixels changed`. Bei Frame 41 wird daraus `0.91 / 255 · 54.62%` – ohne Frame-Präfix. Der Readout unten hat den Frame-Kontext, die Sidebar nicht.
**Why it matters:** Ein Restaurator liest die Zahl als Variantenqualität und vergleicht Varianten bei unterschiedlichen Frames. Das führt zu falschen Schlüssen über „sanfter“ oder „stärker“.
**Fix:** `Frame 41 · …` voranstellen oder einen Clip-Aggregatwert über die Stichproben zeigen; den aktuellen Frame-Wert nur im Inspector führen. Frame-Level nie als Variant-Level präsentieren.
**Suggested command:** `clarify`

### [P1] „100%“ ist 38 % der Quelle, und 400 % hat keinen Pan
**What:** Bei 100 % wird ein 960×720-Still in einer 364×290-Box gezeigt – 37,9 % der nativen Auflösung. Bei 400 % skaliert CSS 4× in dieselbe Box mit `overflow:hidden`; nur die Mitte ist sichtbar, die Ränder sind unerreichbar. Der Wipe (898×673) verhält sich genauso.
**Why it matters:** Für eine Korn- und Detailentscheidung ist „100%“ schlicht falsch, und die Ränder des Frames sind nicht prüfbar. Gleichzeitig bekommt die komprimierte A/B/C-Preview den größten Platz und die exakte Review nur eine 17-px-h3.
**Fix:** Echtes 1:1 mit Pan/Scroll anbieten, effektive Skala anzeigen (`Fit · 100% (1:1) · 200% · 400%`), einen Zoom für Stills und Wipe. Das Ergebnis-Still nach vorn, A/B/C als kompakte, klar als lossy markierte Preview.
**Suggested command:** `optimize`

### [P1] Die primäre Aktion liegt in einem verschachtelten Scroll, zwei Teal-Buttons konkurrieren
**What:** Bei 1600×1000 hat die Sidebar 844 px sichtbare Höhe, aber 959 px Inhalt; `Run Remove Noise` sitzt bei y≈1035–1068, unterhalb der Sidebar-Kante bei 998. Die Rail scrollt unabhängig, die Seite scrollt daneben. Bei nicht aktueller Variante sind „Use as stage output“ und „Run Remove Noise“ beide aktiv und teal, mit unterschiedlicher Semantik.
**Why it matters:** Die wichtigste Aktion ist nicht sichtbar, und zwei Primaries bedeuten keine Primary. „Current stage output“ ist ein Status, der wie ein Button aussieht.
**Fix:** Einen festen Action-Footer in der Rail pinnen: „Apply variant“ als einzige Primary, „Re-render“ sekundär, Stop/Command tertiär. Den Stage-Output als Badge auf Zelle und Panel zeigen, nicht als disabled Button. Wenn die Rail scrollen muss, den Footer außerhalb des Scrollbereichs platzieren.
**Suggested command:** `layout`

### [P2] Der Tab wiederholt einen Frame bis 2.741 px
**What:** Derselbe Frame erscheint in A/B/C, zwei Exact Stills, verstärkter Differenz und einem 900×673-Wipe. Der Wipe wiederholt links/rechts sechsmal: Summary, Hint, Live-Prozent, Label, Status und Grip. „Current stage output“ steht dreimal, der Output-Pfad zweimal.
**Why it matters:** Wiederholung erhöht die Sicherheit nicht, sie begräbt die Entscheidung. Der Page-Height ist 2.741 px bei 1600×1000 und 5.241 px bei 390.
**Fix:** Wipe standardmäßig zuklappen oder zum Hero machen; doppelte Hints/Status/Labels entfernen; Output Path in das Selected-Panel mergen; die Review auf RESULT + Wipe + optionale Differenz mit einer Zoom-Gruppe reduzieren.
**Suggested command:** `distill`

## Persona Red Flags

**Alex (Power User):** Die Wahl von C kann „Use as stage output“ umhängen, während B die zentrale Vergleichsspalte bleibt – er kann dem Commit nach einem Drei-Wege-Vergleich nicht trauen. Keine per-Variante-Metrik-Tabelle, kein Compare-all; das Batch-Grid ist der einzige Neun-Wege-Vergleich und liegt zugeklappt. Der Chart hat zwar einen Tab-Stop, aber keinen Arrow-Handler. Ein Variantenwechsel baut die A/B/C-Player neu und setzt den Playhead auf 0.00; nur die Exact Review behält den Frame.

**Sam (Accessibility/Keyboard):** Off liegt außerhalb der Radiogroup und nutzt `aria-pressed` – eine Auswahl mit zwei Tab-Stops und zwei Semantiken. Der Chart hat 72 Balken à 15 px (bei 1600), weit unter dem 24×24-Ziel; bei 390 sind sie 3,95 px breit. Die verstärkte Differenz kodiert heller/dunkler nur über Teal/Orange; die Caption erklärt es, aber es gibt keine nicht-farbliche Kodierung. Deutsche Fehlertexte aus `review-contrast.js`/`frame-inspector.js` können im englischen Review auftauchen.

**Marta (Archive Restorer):** „100%“ ist 38 % der Quelle; 400 % ist mittig beschnitten ohne Pan, also keine Kornprüfung an den Rändern. Die Sidebar-Zahl ist Frame 1, nicht die Variante. Die A/B/C-Preview ist H.264, und der A–B-Diff enthält Kompressionsartefakte; nur die untere Karte gibt das zu. „Start time (s)“ ist kein Timecode; die Matrix sagt nichts darüber, welche Methode Korn oder Detail schont. Sie wird den Wipe nutzen und dann ZIP oder CLI bemühen.

## Minor Observations

- **A–B-Checkbox layout bug:** Das `<input type=checkbox>` ist **242×13 px** breit, der Labeltext steht darunter (242×38). Ursache: globales `input{width:100%}`; `.denoise-compare-actions` ist nicht von `.checks input{width:auto}` abgedeckt.
- **Chart-Balken:** 72 × 15 px bei 1600; unter dem 24×24-Target. Roving-Tabindex ist korrekt (1 Tab-Stop), aber Arrow-Keys fehlen.
- **Heading-Hierarchie:** Kein `h1`; h3-Größen 15 / 16,38 / 17 px. Die Exact Review ist eine 17-px-h3 unter der Preview.
- **Default-UA-Controls:** 12 Runs bei 13,333 px („Run Remove Noise“, „Play all“, „Previous frame“, „−10“), während Nachbarn 12 px sind. `button, input, select, textarea {font:inherit}` würde das vereinheitlichen.
- **A/C-Layout:** „Show A–B pixel differences“ diffed weiter das versteckte B-Video; die Checkbox sollte dem sichtbaren Paar folgen oder in A/C deaktiviert sein.
- **Playhead:** Ein Variantenwechsel setzt die A/B/C-Player auf 0.00 zurück; nur die Exact Review behält den Frame.
- **9/9:** „Render all 9 variants“ bleibt aktiv und primär, obwohl alle neun fertig sind. „Render missing (0)“ oder disabled wäre ehrlicher.
- **Reihenfolge:** Dropdowns sortieren BM3D → HQDN3D → NLMeans und Balanced → Gentle → Strong; die Matrix sortiert HQDN3D → NLMeans → BM3D und Gentle → Balanced → Strong. Eine Ordnung wählen.
- **Original-Option:** Zeigt einen abgeschnittenen Windows-Pfad. Besser `Original · Neighbors_72.mkv (960×720)`.
- **„Apply to all three“** ist redundant, weil Start/Dauer global sind; „View“ wirkt sofort, die Zeitwerte brauchen Apply. „Reload clips“ oder Apply-on-change.
- **1000 px:** Zwischen Help-Text und „9/9 rendered“ liegen 664 px; die Matrixzellen werden 290×40 mit nur „Ready“, der Vergleich beginnt bei y≈1055 – unter dem ersten Screen.
- **390 px:** Die Tabbar ist 557 px breit in 390 px Viewport; 167 px sind ohne sichtbare Scroll-Affordance verborgen. Die Seite ist 5.241 px lang, drei 350×450-Stills stapeln sich.
- **Chart-Normalisierung:** Jede Variante wird auf ihren eigenen Peak normalisiert, ohne Y-Achse; Balkenhöhen sind zwischen Varianten nicht vergleichbar. Bei gesampelten Filmen suggeriert „Frame changes“ eine Kontinuität, die die Stichproben nicht haben.
- **Deutsche Fehlertexte:** `Ungültige Differenzdaten oder Kontrasteinstellungen.` und `Bildgrößen stimmen nicht überein.` können im englischen Tab auftauchen.

## Questions to Consider

1. Wenn die Exact Review die Wahrheit ist, warum ist sie eine 17-px-h3 unter einer Preview, die die Seite selbst „approximate“ nennt?
2. Was bedeutet „selected“, wenn Matrix-Ring, B, C, Review-Label und Stage-Output auseinanderlaufen dürfen? Welche Instanz ist die Quelle der Wahrheit?
3. Wenn „100%“ 38 % der Quelle ist, was beurteilt der Restaurator dann eigentlich? Würde echtes 1:1 mit Pan die Methodenwahl stärker verändern als jeder Copy-Fix?
4. Warum zeigt die Matrix neun „Ready“-Labels, aber keine einzige Qualitäts-, Geschwindigkeits- oder Kornzahl? Ist sie eine Entscheidungshilfe oder eine Kompatibilitätstabelle für die CLI?
5. Wenn das Batch-ZIP der einzige echte Neun-Wege-Vergleich ist, warum liegt es zugeklappt unter „Batch export“? Ist die beste Evidenz versteckt, weil sie teuer ist – oder weil sie nicht überzeugt?
6. Die Copy sagt viermal „not proof of noise removal“. Welche Evidenz würde sie eigentlich erbringen – ein Korn-/Detailmaß, ein Textur-Crop, ein Gesichts-Close-up oder noch eine verstärkte Differenzkarte?
