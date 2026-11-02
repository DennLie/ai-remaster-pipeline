# Schritt 5: Ursachenanalyse von Kratzermasken

**Status:** Forschungsdiagnose; keine Änderung an der produktiven Kratzenerkennung, keine KI-Gewichte, keine Aufnahme in `main`.

## Ausgangslage aus dem echten Dreifilm-Pilot

[GitHub Actions #41374986827](https://github.com/DennLie/ai-remaster-pipeline/actions/runs/41374986827) führte tatsächlich drei unabhängige historische Filmtransfers durch das ARP-Schritt-1–5-CPU-Experiment. Bei **künstlich eingezeichneten Kratzern** lieferte `cleanup_video.ai_scratch_mask` bei `sensitivity=0.65` folgende pixelweisen F1-Werte:

| Film | Pixel-F1 | Fehlmarkierte Pixel |
|---|---:|---:|
| Lumière – Zug | 0,000 | 246.154 |
| The Great Train Robbery | 0,000 | 1.537.530 |
| Méliès – Mondreise | 0,064 | 876.063 |

Der darauf folgende, ausdrücklich **klassische OpenCV-Telea-Inpainting-Kontrolllauf** verschlechterte die Referenz-PSNR in allen drei Filmen. Das ist **keine Messung des DART-/ProPainter-Modells**.

## Was dieses Experiment zusätzlich misst

Neues Skript: `scripts/diagnose_scratch_masks.py`

1. Prüft **exakte Referenz- und Input-SHA-256** sowie erwartete FPS, Breite, Höhe und Framezahl.
2. Berechnet echte Detektormasken für ein **vorab festes Sensitivitätsraster** von 0.25, 0.45, 0.65 und 0.85. Das Raster ist keine automatische Preset-Optimierung.
3. Ermittelt für jedes Video und jeden Frame **TP, FP, FN, TN, Precision, Recall, F1, IoU, Maskendeckung** und Fehlalarmanteil außerhalb der künstlichen Kratzer.
4. Führt einen **Negativ-Kontrolllauf auf der Ausgangsdigitalisierung ohne zusätzliche synthetische Kratzer** durch. Die dabei ausgewiesene Detektoraktivität ist **nicht** gleichbedeutend mit echten Fehlalarmen, weil die Originaldigitalisierung bereits natürliche Schäden enthalten kann.
5. Führt **zweimal denselben Telea-Inpainting-Algorithmus** aus: einmal auf der vorhergesagten Maske und einmal auf der **nur aus dem Versuch bekannten Ground-Truth-Maske**. Die Differenz hilft, Probleme der Maskenerkennung von Problemen der Inpainting-Interpolation zu trennen. **Die Ground-Truth-Maske darf niemals als normal verfügbares KI-Ergebnis dargestellt werden.**
6. Erzeugt für drei Originalframes pro Szene visuelle Kontaktbilder: Original / künstlicher Schaden / echte Maske / Detektorausgabe mit **grünen True Positives, roten False Positives und blauen False Negatives**.
7. Erfasst rohe Ergebnisse und Hashes in `scratch_diagnostics.json`. Es gibt weder einen automatischen Qualitätsgewinner noch Änderungen am Standard-Sensitivitätswert.

## So wird ein Experiment reproduziert

Zuerst genau den bereits ausgewählten digitalen Referenzclip künstlich beschädigen, beispielsweise den ersten Lumière-Ausschnitt:

```bash
python scripts/restoration_evaluation.py synthesize \
  --source benchmarks/public_domain_films/generated/lumiere_train/approach_probe/original_native.mkv \
  --film lumiere_train --scene approach_probe \
  --kind vertical_scratches --preset gentle --seed 2026 \
  --output-dir benchmarks/evaluation/lumiere_synthetic_scratches

python scripts/diagnose_scratch_masks.py \
  --manifest benchmarks/evaluation/lumiere_synthetic_scratches/manifest.json \
  --sensitivities 0.25,0.45,0.65,0.85 \
  --output-dir benchmarks/evaluation/lumiere_mask_diagnostics
```

Der GitHub-Workflow `.github/workflows/scratch-mask-diagnostics.yml` verarbeitet denselben Film und zusätzlich *The Great Train Robbery* und *Le Voyage dans la Lune* aus dem SHA-gesicherten Commons-Katalog.

Je Film entsteht ein separates, **30 Tage** aufbewahrtes GitHub-Actions-Artefakt `scratch-diagnostic-<film-id>` mit JSON und PNG-Diagnosen – **ohne vollständigen Film**, ohne KI-Gewichte und ohne Audio.

## Wissenschaftliche Grenzen

**Das Sensitivitätsraster ist eine diagnostische Fehleranalyse auf dem vorhandenen Testset.** Wenn wir nach Betrachtung der F1-Werte eine neue Einstellung oder Methode entwickeln, müssen wir ihre Leistungsfähigkeit an **anderen, bisher nicht für das Tuning verwendeten Filmen oder Szenen** prüfen. Es wäre wissenschaftlich unzulässig, den besten Schwellenwert auf diesen drei Filmen zu suchen und dann einen Gewinn auf denselben Filmen als unabhängige Bestätigung zu verkaufen.

Vertikale 1–5-Pixel-Kratzer in digitalen Frames sind nicht repräsentativ für alle realen Filmrisse, Staubpunkte, flächigen Emulsionsschäden, bewegten Kopierdefekte oder Kratzer der Originalnegative. Der Maskendetektor kann sehr dünne vertikale Konturen im natürlichen Bild fälschlich als Kratzer interpretieren. Für echte Filmrestaurierung brauchen wir zusätzlich manuell annotierte Fehlerkarten, langfristige temporale Konsistenz, verschiedene Defekttypen und negatives Kontrollmaterial (Bäume, Zäune, Fensterlinien, Zwischentitel).

**In diesem PR wird der Produktionsdetektor bewusst nicht verändert.** Erst nach Fehlerdiagnose, reproduzierbaren Kontrollen und separatem Holdout-Versuch sollten wir einen verbesserten Modus in ARP anbieten. Die Lizenzrechte an einer bestimmten Digitalisierung und alle Modellgewichte bleiben eine separate Prüfung.
