# Automatische wissenschaftliche Evaluation der ARP-Schritte 1–5

**Status:** CPU-Forschungsadapter, Draft PR, November 2026. Kein Merge nach main, keine automatische Freigabe eines Modells oder Filmmaterials.

Mit `scripts/restoration_evaluation.py` (PR #19) können wir aus einem
existierenden digitalisierten Filmclip eine **bekannte, künstliche
Degradation** und dazugehörige Referenz erzeugen. Dieser PR
schließt die bisher offenen CPU-Adapter für Schritte 1, 2, 4 und 5 an.
Die automatische klassische Detail-Enhancement-Matrix für **Schritt 3**
ist bereits in `scripts/run_scientific_classical_grid.py` vorhanden.

## Vollständiger Status

| Schritt | Tatsächlich automatisiert | Eingangsartefakt | Zusätzliche Aufgabe | Nicht behauptet |
|---|---|---|---|---|
| 1 Light Deflicker | **Exakter konservativer PR-#6-CPU-Algorithmus** `film_deflicker.py`, mit Voranalyse, Gain-Caps und Cut-Detektion | `temporal_flicker` | PSNR/SSIM gegenüber Referenz, zeitlicher Referenzfehler, geänderte Gain-Anzahl | Keine universelle Deflicker-Verbesserung echter Filme |
| 2 Remove Noise | **Echte** ARP-Filter `off, hqdn3d, nlmeans, bm3d`, je nach Preset | `read_noise` | PSNR/SSIM und zeitlicher Referenzfehler | Kein erhaltener chemischer Filmkornnachweis |
| 3 Detail Enhancement | Bereits vorhanden: `off, unsharp, cas, multiscale, richardson_lucy` | `gaussian_blur` | PSNR/SSIM, kontrollierte PSF | Kein Beleg authentischer wiedergewonnener Details |
| 4 Stabilization | **Echtes** ARP `stabilize_video.py` mit `libvidstab` in Single-Shot-Konfiguration | **Neu:** `camera_jitter` mit reproduzierbarer ±1,5/3/5-px-Translation | PSNR/SSIM und residualer Bildversatz per zentralem Hanning-Phasenkorrelations-Test | Keine fertige Szenenschnitt-/Dolly-/Verdeckungsmetrik |
| 5 Dust/Scratches | **Echte** ARP CPU-Maskenerkennung `cleanup_video.ai_scratch_mask`; zusätzlich explizit benannter **OpenCV-Telea-Baseline** (kein KI-Inpainting) | `vertical_scratches` plus binäre Pixel-GT-Masken | Precision/Recall/F1/IoU und Reparatur-PSNR/SSIM, Maskenfehler | **ProPainter/DART-KI wurde NICHT ausgeführt** |

Der Code von `film_deflicker.py` ist eine identische Kopie aus
[Draft-PR #6](https://github.com/DennLie/ai-remaster-pipeline/pull/6),
weil dieses Feature auf einem anderen PR-Stapel liegt. Für spätere
Integration muss es **zusammengeführt und dedupliziert** werden. Die
Kopie ist nur für die wissenschaftliche Evaluationsbranch vorhanden.
Alle Modellgewichte und GPU-Backends bleiben unangetastet.

## Beispiel: fünf unabhängige wissenschaftliche Auswertungen

Jede Messung verwendet den **gleichen ursprünglichen FFV1-Filmclip**,
erzeugt aber einen eigenen bekannten Fehlerzustand.
Für einen realen Clip ist der hier verwendete Pfad gegebenenfalls
durch eine andere Film-/Szene-ID aus PR #18 zu ersetzen.

```bash
SOURCE="benchmarks/public_domain_films/generated/caligari/portrait_probe/original_native.mkv"
mkdir -p benchmarks/evaluation

# Schritt 1: künstliches periodisches Flackern.
python scripts/restoration_evaluation.py synthesize \
  --source "$SOURCE" --film caligari --scene portrait_probe \
  --kind temporal_flicker --preset balanced \
  --output-dir benchmarks/evaluation/e1_flicker

python scripts/evaluate_steps_1to5.py \
  --stage deflicker \
  --manifest benchmarks/evaluation/e1_flicker/manifest.json \
  --output-dir benchmarks/evaluation/e1_result

# Schritt 2: künstliches Leserauschen, 4 reale ARP-Methoden.
python scripts/restoration_evaluation.py synthesize \
  --source "$SOURCE" --film caligari --scene portrait_probe \
  --kind read_noise --preset balanced \
  --output-dir benchmarks/evaluation/e2_noise

python scripts/evaluate_steps_1to5.py \
  --stage denoise \
  --manifest benchmarks/evaluation/e2_noise/manifest.json \
  --methods off,hqdn3d,nlmeans,bm3d --presets gentle,balanced,strong \
  --output-dir benchmarks/evaluation/e2_result

# Schritt 3: die bereits vorhandene Vollmatrix.
python scripts/restoration_evaluation.py synthesize \
  --source "$SOURCE" --film caligari --scene portrait_probe \
  --kind gaussian_blur --preset balanced \
  --output-dir benchmarks/evaluation/e3_blur

python scripts/run_scientific_classical_grid.py \
  --manifest benchmarks/evaluation/e3_blur/manifest.json \
  --methods off,unsharp,cas,multiscale,richardson_lucy \
  --presets gentle,balanced,strong \
  --output-dir benchmarks/evaluation/e3_result

# Schritt 4: kontrolliertes jitter mit bekannter sauberer Vorlage.
python scripts/restoration_evaluation.py synthesize \
  --source "$SOURCE" --film caligari --scene portrait_probe \
  --kind camera_jitter --preset balanced \
  --output-dir benchmarks/evaluation/e4_jitter

python scripts/evaluate_steps_1to5.py \
  --stage stabilize \
  --manifest benchmarks/evaluation/e4_jitter/manifest.json \
  --output-dir benchmarks/evaluation/e4_result

# Schritt 5: künstliche Kratzer plus exakt bekannte binäre Masken.
python scripts/restoration_evaluation.py synthesize \
  --source "$SOURCE" --film caligari --scene portrait_probe \
  --kind vertical_scratches --preset balanced \
  --output-dir benchmarks/evaluation/e5_scratches

python scripts/evaluate_steps_1to5.py \
  --stage scratch_masks \
  --manifest benchmarks/evaluation/e5_scratches/manifest.json \
  --scratch-sensitivity 0.65 \
  --output-dir benchmarks/evaluation/e5_result
```

Jeder Ergebnisordner schreibt `stage_report.json`, die getrennte
`scores/evaluation.json`, Kandidatenvideos und Sidecars. Der Schritt
5 erzeugt außerdem PNG-Masken im Ordner `predicted_masks/`.

## Auswertung nach Film statt nach Einzelbildern

Die bestehenden Film- und Szenenaggregationen sind in
`scripts/restoration_evaluation.py summarize` implementiert.
Ergebnisse desselben Algorithmus müssen pro Schadenstyp aggregiert
werden, nicht über unterschiedliche Schadensarten hinweg. Die
Software kann **negative** Ergebnisse und leere
Maskenfunde ausdrücklich melden: Precision/Recall/F1 werden
beispielsweise null, wenn die ARP-Maskenerkennung auf dem
synthetischen Beispiel keine Schäden findet.

Eine **statistische Film-Analyse** erfordert mehrere voneinander
unabhängige Filme und länger gewählte Szenen, nicht nur die
24 Testframes einer Kurzsequenz. Ein Szenen-/Film-Bootstrap mit
drei Filmen bleibt ein Pilot mit sehr breiten Unsicherheiten.

## Schutz vor methodisch falschen Schlussfolgerungen

- Messwerte beziehen sich auf die **rekonstruierte vorhandene
  Digitalisierung**, nicht auf das vermeintlich unbeschädigte
  historische Filmnegativ.
- Das synthetische Rauschen ist **homoskedastisches Read Noise**,
  kein realistisches photochemisches Filmkorn; die
  Kamerabewegung sind einfache Translationen, keine
  echte Filmprojektor-Mechanik; Kratzer sind vereinfachte
  vertikale Pixelstreifen.
- Die Referenzdatei wird beim Testen niemals als Modellinput
  verwendet. Alle Methoden erhalten **nur das künstlich
  beschädigte Video**.
- Schritt 4 nutzt ein **orientierendes** bewegungskompensiertes
  Maß per Phasenkorrelation. Bei zu geringer
  Übereinstimmungs-Sicherheit wird `INSUFFICIENT_PHASE_CORRELATION`
  gemeldet, **kein falscher genauer Wert**.
- Die Schritte 1 und 5 haben Einschränkungen: Einfache
  Lichtschwankungen/Staublinien sind nicht repräsentativ für
  authentische alternde Filmkopien. Ein gutes Score belegt
  keine bessere filmarchivarische Originaltreue.
- Bei Schritt 5 ist der maskenbasierte Reparaturclip eine
  **OpenCV-Telea-Kontrollbaseline**. Die tatsächliche
  KI-ProPainter-Rekonstruktion benötigt die
  entsprechende (nicht automatisch installierte) GPU-Umgebung,
  Modellgewichte und gesonderte Lizenzprüfung.
- Alle genannten Ergebnisse sind **nur CPU-Automation**,
  kein RTX-4090-, NAFNet-, Restormer- oder FlashVSR+-Test.
