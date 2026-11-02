# Wissenschaftliche Evaluationsadapter für ARP-Schritte 1, 2, 4 und 5

**Status:** CPU-Forschungsadapter auf dem vorhandenen Evaluationsframework aus Draft-PR #19; keine Änderungen an `main`, keine neuen KI-Modellgewichte oder kommerziellen Freigaben. Das ist eine technische Versuchsanordnung, **keine** Aussage zur historischen Originaltreue.

Das Framework erwartet zunächst ein **kurzes, verifiziertes, unverändertes** Video als digitale Ausgangsreferenz. Die synthetische Schädigung wird aus genau dieser Referenz abgeleitet. Weder die digitale Kopie noch ihre Pixel sind automatisch das authentische historische Filmoriginal.

## Schritt 1 – Light Deflicker

Im untersuchten ARP-Fork gibt es zurzeit keinen eigenen Deflicker-Backend. Der Forschungsadapter verwendet deshalb FFmpegs dokumentierten Filter `deflicker` mit `am` (arithmetisches Mittel) oder `median`, Fenstergrößen 3/5/9 (Gentle/Balanced/Strong). Er bearbeitet **ausschließlich** einen bewusst synthetisch flackernden Clip; es gibt keine automatische Änderung an der produktiven Pipeline.

```bash
python scripts/restoration_evaluation.py synthesize \
  --source benchmarks/public_domain_films/generated/caligari/portrait_probe/original_native.mkv \
  --film caligari --scene portrait_probe \
  --kind temporal_flicker --preset balanced \
  --output-dir benchmarks/evaluation/deflicker_caligari

python scripts/scientific_stage_adapters.py grid \
  --manifest benchmarks/evaluation/deflicker_caligari/manifest.json \
  --methods off,ffmpeg_deflicker --presets gentle,balanced,strong \
  --output-dir benchmarks/evaluation/deflicker_caligari_grid
```

Die Ausgabe enthält für jeden Kandidaten eine verlustfreie FFV1-Datei, SHA-256-Herkunftssignatur und echte Full-Reference-PSNR/SSIM/zeitliche Referenzabweichung gegenüber einer sauberen **digitalen** Referenz. Das Filter kann beabsichtigte Belichtungsveränderungen, Szenenschnitte und Schwarzblenden falsch korrigieren; diese sind gesonderte Negativkontrollen.

## Schritt 2 – Remove Noise

Hier verwendet der Adapter die **tatsächlichen** Backends `denoise_video.run` aus ARP: HQDN3D, NLMeans, BM3D und Off. Keine Mock-Implementierung. Bei Auswahl von BM3D muss die lokal verwendete FFmpeg-Version den Filter bereitstellen, sonst scheitert dieser Kandidat explizit. Der bisherige Testschaden `read_noise` ist bewusst einfaches homogenes Sensorrauschen, **kein** authentisches analoges Filmkorn.

```bash
python scripts/restoration_evaluation.py synthesize \
  --source benchmarks/public_domain_films/generated/caligari/portrait_probe/original_native.mkv \
  --film caligari --scene portrait_probe \
  --kind read_noise --preset balanced \
  --output-dir benchmarks/evaluation/denoise_caligari

python scripts/scientific_stage_adapters.py grid \
  --manifest benchmarks/evaluation/denoise_caligari/manifest.json \
  --methods off,hqdn3d,nlmeans,bm3d \
  --presets gentle,balanced,strong \
  --output-dir benchmarks/evaluation/denoise_caligari_grid
```

Sämtliche Algorithmen nutzen den **exakt gleichen beschädigten Input**, werden nicht stillschweigend per unterschiedlichem Seeking auf andere Bilder gesetzt und müssen Bildzahl, FPS sowie Auflösung unverändert lassen.

## Schritt 4 – Camera Stabilisation

`scientific_stabilization_eval.py prepare` erzeugt eine deterministische, frameweise bekannte **Translationsstörung** mit 1,5/3/5 Pixel maximalem Shift pro Achse und eine 1×-Referenz. Der Versuchsaufbau beschränkt sich bewusst auf Translationen; Rotationsfehler, Kameraschwenks, Rolling Shutter, Zoom, Bildverlust und echte verlorene Frames sind noch nicht simuliert.

```bash
python scripts/scientific_stabilization_eval.py prepare \
  --source benchmarks/public_domain_films/generated/caligari/portrait_probe/original_native.mkv \
  --film caligari --scene portrait_probe --preset balanced \
  --output-dir benchmarks/evaluation/jitter_caligari

# Realen ARP-libvidstab-Stabilisierer ausführen und gegenüber der
# gleichen Referenz auswerten; keine automatisch vorausgesetzte Verbesserung.
python scripts/scientific_stabilization_eval.py run-arp \
  --manifest benchmarks/evaluation/jitter_caligari/manifest.json \
  --smoothing 5 --zoom 1 \
  --output-dir benchmarks/evaluation/jitter_caligari_arp
```

Gemessen werden der zentrale ROI-Pixelfehler und per Phase-Correlation geschätzte **Restverschiebung** gegenüber der digitalen Referenz. Quelle, Verschiebungsvektoren und Prüfsummen liegen im Manifest. Die Auswertung meldet die Größe des nicht betrachteten Randbereichs. **Der Phasenkorrelationswert ist ein Proxy**, der bei echten Bewegungen, flächigen Bildern, Zoom/Crop, veränderten Farben und wechselnden Inhalten unzuverlässig sein kann. Er ist kein Beleg für hochwertige Filmstabilisierung. Das ARP-Modul muss für `run-arp` FFmpeg mit `vidstabdetect` und `vidstabtransform` enthalten.

## Schritt 5 – Remove Dust & Scratches

Der Test ruft `cleanup_video.ai_scratch_mask` aus dem realen ARP-Quellcode auf. Synthetisch eingezeichnete schmale vertikale Kratzer besitzen exakt bekannte pixelweise GT-Masken. Für den Detektor werden True Positives, False Positives, False Negatives, Precision, Recall, F1 und maskierte Flächenanteile ausgegeben.

```bash
python scripts/restoration_evaluation.py synthesize \
  --source benchmarks/public_domain_films/generated/caligari/portrait_probe/original_native.mkv \
  --film caligari --scene portrait_probe \
  --kind vertical_scratches --preset balanced \
  --output-dir benchmarks/evaluation/scratches_caligari

python scripts/scientific_stage_adapters.py scratch-mask \
  --manifest benchmarks/evaluation/scratches_caligari/manifest.json \
  --sensitivity 0.65 \
  --output-dir benchmarks/evaluation/scratch_mask_caligari
```

**Nicht** gemessen werden hier: ProPainter-Inpainting, DART-Gesamtrestaurierung, Staub-/Fleckenmodelle oder maskierte Detailtreue von echten historischen Filmdefekten. Diese folgen als eigene Messmodule. Der einfache Kratzengenerator ist derzeit kein realistischer Ersatz für echte beschädigte Scans.

## Was danach noch erforderlich ist

1. Testmatrix nicht nur auf einer Szene, sondern für jede freigegebene Szene auf **mehreren voneinander unabhängigen Filmen** reproduzierbar ausführen. Einstellungen nicht auf Testfilmen nachoptimieren.
2. Für jeden Teilbereich spezifische bekannte Schäden und Negativkontrollen ergänzen, etwa Fades (Deflicker), authentische Kornstatistik (Denoise), Kameraschwenks/Rotation/äußeren Bildverlust (Stabilisierung) und nichtvertikale Kratzer sowie Staub (DART).
3. Auswertungen pro Szene, danach gleichgewichtet pro Film aggregieren; Unsicherheit und stärkste Ausreißer berichten. Keine globale 1–5-Rangliste.
4. Ein echtes GPU- und qualitätsgesichertes KI-Reparaturexperiment getrennt von diesen CPU-Tests durchführen und das kommerzielle Lizenzregime prüfen.

**Quellen:** FFmpeg `deflicker`, `hqdn3d`, `nlmeans`, `bm3d` und `vidstabdetect`/`vidstabtransform`: https://ffmpeg.org/ffmpeg-filters.html

Die getesteten historischen Digitalisierungen bleiben nur Versuchsmaterial; auch gemeinfreie Filmwerke rechtfertigen keine pauschale Freigabe jeder Digitalisierung, Restaurierung oder neueren Tonspur.
