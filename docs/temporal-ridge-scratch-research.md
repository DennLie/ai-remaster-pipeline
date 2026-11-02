# Schritt 5: neue Forschungsvariante für Kratzenerkennung mit temporaler Referenz

**Stand:** 2026-11-02 · Forschung, keine produktive Aktivierung · auf [PR #23](https://github.com/DennLie/ai-remaster-pipeline/pull/23) aufbauend.

## Motivation

Die vorherige Referenzdiagnose zeigte auf drei kurzen historischen Digitalisierungen einen schlechten Pixel-F1 für synthetische vertikale Kratzer. Das bloße Deaktivieren der Ganzspaltenextrapolation senkte zwar die Maskenfläche, verfehlte aber **alle** Ground-Truth-Kratzer. Das war **keine Reparatur**.

In diesem PR wird bewusst **ein eigener, isolierter Forschungsdetektor** getestet. Die bisherige `cleanup_video.ai_scratch_mask` bleibt als Baseline und standardmäßig unverändert.

## Unabhängige Merkmale

Die neue Funktion `scripts/temporal_ridge_scratch_research.py:detect_sequence` erhält **ausschließlich die beschädigten Videoframes**, keine saubere Referenzdatei und keine künstlichen Ground-Truth-Kratzer. Drei konservative Bedingungen müssen erfüllt sein:

1. **Ridge statt Kante:** Ein Pixelband muss sich in derselben Richtung von der linken **und rechten** Nachbarschaft unterscheiden. Ein reiner Helligkeitssprung (z. B. Fensterschatten) ist keine beidseitige Linie.
2. **Temporale Neuartigkeit:** Das Pixel muss deutlich vom robusten 20%- bzw. 80%-Quantil der zeitlichen Einzelpixelverteilung abweichen. Über mehrere Bilder identische Rahmenlinien werden so häufig zurückgewiesen. Nachteil: **fixierte reale Kratzer** und starke Kamerabewegung können problematisch sein.
3. **Geometrische Sicherheit:** Vertikale Mindestlänge, maximale Komponentenbreite, keine vollständige Spaltenextrapolation, maximal **3 % Bilddeckung**, andernfalls verwirft der Forschungsdetektor die gesamte Maske für den betroffenen Frame. Das ist konservativ und verhindert überbreite KI-Reparaturforderungen.

**Die Hyperparameter und der Sensitivitätswert 0,65 sind vor dem unabhängigen Holdout festgelegt.** Es erfolgt keine Optimierung anhand der Holdout-Ergebnisse. Eine eventuelle spätere Anpassung erfordert **andere** Filme/Szenen.

## Wissenschaftlicher Vergleich

Das GitHub-Workflow `.github/workflows/temporal-ridge-scratch-research.yml` verwendet separate Gruppen:

- **Entwicklung / bisher bereits evaluierte Szenen:** Lumière `approach_probe`, *Great Train Robbery* `interior_action_probe`, Méliès `stage_probe`.
- **Unabhängiger Film-Holdout (für diesen Detektor nicht beim Entwickeln verwendet):** *Caligari* `expressionist_set_probe`, *Gertie* `dinosaur_motion_probe`.

Die Definitionen sind hart in den GitHub-Actions-Jobs hinterlegt. Die beiden Holdout-Filme laufen erst nach erfolgreichen Entwicklungstests. Jeder Film erhält **synthetische, identische, exakt maskierte künstliche Kratzer** aus derselben wissenschaftlichen Schadenssimulation, **keine** neue Inhaltsauswahl. Der SHA-256 der übertragenen Quelle muss mit dem kuratierten Commons-Katalog übereinstimmen.

Für beide Verfahren werden identisch berichtet:

- Pixelgenaue True Positives / False Positives / False Negatives / True Negatives, Precision, Recall, F1 und IoU.
- Negative-Control-Maskierungsfläche auf der unveränderten Digitalisierung. Das ist **keine** echte False-Positive-Rate für historisches Filmoriginal, da historische Kratzer schon darin vorhanden sein können.
- Fehler durch eine **klassische OpenCV-Telea-Reparatur** unter der Kandidatenmaske, getrennt *innerhalb* und *außerhalb* der bekannten künstlichen Kratzer.
- Kontaktbilder Original / künstliche Schäden / synthetische Ground-Truth-Maske / Kandidatenmaske mit Grün=TP, Rot=FP, Blau=FN.
- Quellenhash, Filmtitel, Szenen-ID, Split, Framezahl und feste Parameter in `comparison.json`.

Ein durchschnittlich niedrigerer Maskierungsanteil ist **kein Beleg** für bessere Erkennung; bei Null-Recall wird das Verfahren auch dann verworfen.

## Lokal ausprobieren

```bash
# Erst mit vorhandener digitaler Filmreferenz eine kontrollierte Testbedingung aufbauen.
python scripts/restoration_evaluation.py synthesize \
  --source benchmarks/public_domain_films/generated/lumiere_train/approach_probe/original_native.mkv \
  --film lumiere_train --scene approach_probe \
  --kind vertical_scratches --preset gentle --seed 2026 \
  --output-dir benchmarks/evaluation/ridge_lumiere_source

# Vorab fixierte Konfiguration vergleichen.
python scripts/temporal_ridge_scratch_research.py \
  --manifest benchmarks/evaluation/ridge_lumiere_source/manifest.json \
  --split development --sensitivity 0.65 \
  --output-dir benchmarks/evaluation/ridge_lumiere_research
```

**Keine Modellgewichte, keine CUDA-Pflicht, keine Änderungen an GUI oder produktiver Detektionspipeline.** Das Tool schreibt nur JSON und Stichprobenbilder in seinen Ausgabepfad; Quellen bleiben unverändert. Die GitHub-Action lädt vorübergehend die Originaldigitalisierung, speichert aber nur die numerische Diagnose und die Bildstichproben als zeitlich begrenzte Artefakte.

## Grenzen und Entscheidungskriterien

- Der neue Detektor **kann** wiederholte statische Kratzer fälschlich verwerfen, ebenso Kratzer bei Kameraschwenks oder frühen variablen Frame-Raten. Er enthält weder optischen Fluss noch echte Motion Compensation.
- Extrem helle/dunkle Zeichnungen, Trickeffekte, Text, Korn und einzelne schwarze Filmränder erfordern gesonderte Negativkontrollen.
- Kurze 24-Frame-Clips und ein einziger synthetischer Kratzentyp sind nur ein **Pilot**, keine Aussage zur Allgemeingültigkeit.
- Vor Aktivierung sind zusätzlich reale **manuell annotierte** Kratzer und weitere unabhängige Filmquellen nötig.
- Eine verbesserte Referenz-PSNR mit bekannter Maske wäre **keine** Messung von DART-/ProPainter-Inpainting, keine historische Echtheitsprüfung und keine pauschale kommerzielle Nutzungserlaubnis.
- Ein Verfahren wird nur als experimenteller Kandidat weitergeführt, wenn es auf dem **unabhängigen Holdout** sowohl genügend Defekte erkennt **als auch** die Fehlmaskierung gegenüber dem Standard verringert. Ohne Treffer keine automatische „Verbesserung“.

**Erste Entscheidung:** Ergebnisse aller fünf Filme und die Fehlerkarten überprüfen, bevor ein neuer Produktionsmodus überhaupt vorgeschlagen wird. Kein automatisches Merge.
