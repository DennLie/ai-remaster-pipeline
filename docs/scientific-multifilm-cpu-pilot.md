# Multi-Film-Pilot: echte ARP-Schritte 1–5, synthetisch bekannte Fehler

**Forschungsstand 2026-11-02 · CPU-only · Draft-PR · keine Änderungen an `main`**

Dieser Versuch erweitert die erfolgreiche Integrationsprüfung von
[Draft-PR #20](https://github.com/DennLie/ai-remaster-pipeline/pull/20)
auf **drei verschiedene historische Filmwerke**: Lumières
*L'Arrivée d'un train*, Porters *The Great Train Robbery* und Méliès'
*Le Voyage dans la Lune*. Jede Quelle kommt aus dem bereits kuratierten
Commons-Katalog mit separaten Transfer-Hashes.

## Genaues Studiendesign (erster Pilot)

- **Untersuchungseinheit:** zunächst ein **24-Frame-Ausschnitt pro Film**,
  nicht 24 unabhängige Beobachtungen; später mindestens zwei Szenen
  und mehr Filmwerke.
- **Konditionen:** pro Clip dieselben nachweisbar reproduzierbaren
  künstlichen Defekte und immer die unveränderte digitalisierte
  Originaldatei als Vollreferenz. Defekterzeugung ist seedgebunden.
- **Schritt 1:** `temporal_flicker` → **ARP film_deflicker**
  aus dem Forschungszweig (mit wissenschaftlicher Referenzmessung).
- **Schritt 2:** `read_noise` → **Off, HQDN3D, NLMeans und BM3D**.
- **Schritt 3:** `gaussian_blur` → **Off, CAS und Richardson–Lucy**.
- **Schritt 4:** `camera_jitter` → echter
  **FFmpeg/libvidstab**-Stabilisierer, zusätzlich
  markierte Restbewegungs-Proxys.
- **Schritt 5:** `vertical_scratches` → echter
  **ARP-Kratzendetektor mit Precision/Recall/F1**
  und ausdrücklich **klassischer OpenCV-Telea-Kontrollreparatur**,
  *nicht* die KI-DART/ProPainter-Inpaintingqualität.

Alle Algorithmen lesen denselben jeweils künstlich beschädigten
FFV1-Clip, nicht eigenständig gesuchte Zeitmarken in Langfilmen.
Eingabe-, Ausgabe- und Herkunfts-Hashes sowie FPS, Pixelmaße,
Framezahlen und Parameter werden dokumentiert.

## Lokal starten

Vorbereitete Clips unter
`benchmarks/public_domain_films/generated/<film-id>/<scene-id>/`
können verwendet werden. Um ein Quellenpaket zu erstellen:

```bash
python scripts/prepare_public_domain_films.py \
  --film lumiere_train \
  --no-classical \
  --output-dir benchmarks/public_domain_films/generated/lumiere_train
```

Anschließend für alle drei Filme analog.
Wenn ihre Quelle noch vorhanden und alle Rechtehinweise geprüft sind:

```bash
python scripts/run_multifilm_scientific_eval.py \
  --corpus-root benchmarks/public_domain_films/generated \
  --films lumiere_train,great_train_robbery,melies_moon \
  --scenes-per-film 1 \
  --stages deflicker,denoise,detail,stabilize,scratch_masks \
  --seed 2026 \
  --output-dir benchmarks/evaluation/scientific_three_film_pilot
```

Ergebnis:

```text
scientific_three_film_pilot/
  multifilm_report.json     # Übersicht, Film-Hashes, Status aller 15 Stage-Runs,
                            # gleich gewichtete Filmmittel und bootstrap-CIs
  <film-id>/<scene-id>/<stage-name>/
    source/                 # digitaler Referenzclip + synthetischer Schaden
    algorithms/             # tatsächliche ARP-Kandidaten und einzelne Score-JSONs
```

Vorher bestehende Ergebnisordner werden **nicht überschrieben**.
Die vollständigen digitalen Filmmaster werden **nicht**
in Git eingecheckt; GitHub Actions lädt sie nur temporär herunter.
Das Workflow-Artefakt enthält **nur JSON-Berichte**, keine
originalen oder restaurierten Videos.

## Wissenschaftliche Interpretation

`mean_scene_paired_psnr_delta_db` ist der mittlere Unterschied
zum jeweils künstlich beschädigten Eingang bei bekannter digitaler
Referenz. Ein positiver Wert bedeutet bessere Rekonstruktion
**dieser speziellen künstlichen Degradation**, keine allgemeine
archivarische Wahrheit.

Die vorhandene Auswertung mittelt zunächst auf Szenenebene,
dann gleichgewichtet über Filme, und führt einen vorläufigen
hierarchischen Bootstrap durch. **Drei Filme mit je einer
kurzen Szene sind nur ein Pilot**; ein numerisch schmales
Intervall wäre noch kein Beleg für Überlegenheit auf
Filmgeschichte insgesamt.

- **Keine** gemeinsame Zahl für alle fünf Schritte.
- **Keine** automatische subjektive Qualitätsrangliste.
- **Keine** Behauptung, dass FFmpeg-Artefakte oder die historische
  Bildwahrheit objektiv mit diesen Metriken bewertet sind.
- Keine automatische Originaltreue-Erlaubnis für generative
  NAFNet/Restormer/SeedVR2/FlashVSR/FlashVSR+-Modelle.
- Die Verfügbarkeit und kommerzielle Rechte bestimmter
  Digitalisierungstransfers bleiben separat zu prüfen.

Die eigentliche Modellbewertung auf deiner **RTX 4090** folgt
erst nach dieser CPU-Pilotprüfung mit einer festgelegten, nicht
nachoptimierten Auswahl an Testfilmen und Szenen.
