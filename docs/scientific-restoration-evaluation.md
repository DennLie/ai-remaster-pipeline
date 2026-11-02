# Wissenschaftlich überprüfbare ARP-Restaurierungsevaluation — erster Baustein

**Stand: November 2026 · Research / Draft · keine Übernahme nach main.**

Diese erste Stufe ergänzt die bisherigen **beschreibenden** Vergleiche
(Pixeländerung, Laplacian, Clipping) durch echte **Full-Reference-Messungen
bei bekannter kontrollierter Degradation**. Sie ersetzt *nicht*
archivarische Quellenkritik, Rechteprüfung oder verblindete
Qualitätsbeurteilungen.

## Grundannahme

Die sauberste verfügbare Datei aus einem alten Film bleibt nur eine
**digitale Übertragung**, keine authentische, unbeschädigte Kameraaufnahme.
Wir können daraus synthetisch definierte Schäden erzeugen und messen, ob
das Bild im Vergleich zu **dieser Digitalisierung** wiederhergestellt wird.
Das beweist ausdrücklich **nicht**, dass die ursprünglichen verlorenen
Bilddetails des historischen Films zurückgewonnen wurden.

**Zwei getrennte Welten dürfen niemals in einer Qualitätsrangliste
vermengt werden:**

- **Kontrollierte künstliche Schäden**: saubere Referenz aus der
  vorhandenen Digitalisierung, künstlich beschädigter Input und
  Kandidaten-Outputs. Full-reference PSNR/SSIM ist hier angemessen.
- **Unbekannte echte historische Schäden**: vorhandener Original-Scan
  ohne gesichertes Ground Truth. Hier sind PSNR/SSIM gegenüber dem
  beschädigten Original **keine Restaurierungsqualität**.
  Beschreibende Änderungen, Masken, Temporalität und Menschenurteile
  sind getrennt zu dokumentieren.

## Phase 1 (dieser PR): CPU-Kontrollen

Der neue Aufruf ist
\`scripts/restoration_evaluation.py\` mit drei Unterbefehlen:

1. \`synthesize\`: erzeugt aus einem genau ausgewählten nativen FFV1
   Filmclip die Dateien \`reference.mkv\`, \`degraded.mkv\` sowie
   \`manifest.json\`. Die Quelldatei bleibt unberührt.
2. \`evaluate\`: wertet den degradierten Baseline-Input und beliebige
   Kandidatenausgaben gleicher Geometrie/Framezahl/FPS gegen die
   **identische Referenz** aus. Fehlende oder falsche Input-Hashes
   in bekannten Sidecars führen standardmäßig zum Abbruch.
3. \`summarize\`: aggregiert **gepaarte** Unterschiede
   pro Szene und Film. Jeder Film hat gleiches Gewicht.
   Erst ab drei Filmen werden hier vorläufige hierarchische
   Film-/Szenen-Bootstrap-Intervalle berechnet.
   Mit weniger als drei Filmen wird das Ergebnis explizit
   als explorativ ohne CI gekennzeichnet. Drei Filme sind
   immer noch nur ein kleiner Pilot, keine repräsentative
   Population alter Filme.

### Synthetische Schäden

| Typ | Gentle | Balanced | Strong | Grenzen |
|---|---:|---:|---:|---|
| \`gaussian_blur\` | σ 0,65 px | 1,10 px | 1,65 px | Modelliert weder tatsächliche Unschärfefunktion noch authentische Kopierverluste |
| \`read_noise\` | σ 5/255 | 10/255 | 18/255 | Einfaches homogenes Sensorrauschen, **kein** historisch realistisches Filmkorn |
| \`temporal_flicker\` | 5 % | 10 % | 17 % | Periodische synthetische Luminanzänderung, **kein** Modell intentionaler Lichtwechsel |
| \`vertical_scratches\` | 1 Linie | 2 Linien | 4 Linien | Vereinfachte vertikale Kratzer, mit exakten Binärmasken je Frame |

Die Zufallsquelle verwendet den ganzzahligen Seed,
den Filmtitel-/Szenenschlüssel und den Frameindex.
So bleiben Ergebnisse unabhängig von der Job-Reihenfolge
reproduzierbar. Die ausgewerteten Frames werden als
8-Bit-RGB in verlustfreie FFV1-MKV-Zwischenvideos geschrieben.
Sämtliche vorhandenen Tonspuren werden weggelassen.

Alle erzeugten Ausgaben und deren Eingabe-Kryptohashes werden
dokumentiert. Es wird nie eine bestehende Ergebnisordnerstruktur
überschrieben.

### Metriken

| Metrik | Interpretation | Zulässigkeit |
|---|---|---|
| \`psnr_rgb_db\` | PSNR gegenüber bekanntem, dekodiertem RGB8-Referenzvideo | Kontrollierte Degradation (nur dort) |
| \`ssim_rgb\` | SSIM pro Bild, Durchschnitt über Frames; RGB, Gaußfenster, Datenbereich 255 | Kontrollierte Degradation (nur dort) |
| \`mse_rgb_gamma_encoded_0_1_range\` | Mittlerer quadratischer Fehler auf **gamma-kodiertem** RGB8 / 255 | Nicht linear-light oder HDR-fähig |
| \`known_scratch_mask_mse\` | Fehler innerhalb der **künstlichen** Ground-Truth-Kratzerregion | Nur bei synthetischen Kratzern |
| \`temporal_reference_error_rmse\` | Fehler in Bilddifferenzen gegenüber dem bekannten Referenzverlauf | Kein optical-flow-basiertes Video-Konsistenzmaß |
| \`delta_psnr_db_vs_degraded\` | **Paarweise** Veränderung gegenüber der gleich beschädigten Baseline | Höher bedeutet bei *diesem* künstlichen Defekt bessere Referenznähe |
| \`delta_ssim_vs_degraded\` | Paarweise SSIM-Veränderung | Kein archivarischer Gesamtqualitätsscore |

Exakt identische Referenz/Output-Paare haben mathematisch
unendlichen PSNR. Um **gültiges JSON** und stabile Differenzwerte
zu erhalten, werden sehr kleine MSE-Werte auf \`10^-12\`
begrenzt und bei **120 dB** mit
\`psnr_capped_at_120db_for_exact_match=true\`
explizit markiert. Die \`perfect_rgb_match\`-Kennzeichnung
macht echte vollständige Übereinstimmung getrennt sichtbar.

**Noch nicht enthalten:** LPIPS/DISTS (benötigen sorgfältig geprüfte
Modellgewichte und Versionsangaben), optischer Fluss mit
Verdeckungsmasken, No-reference-Halluzinationsprüfung,
Masken-Precision/Recall/F1, GPU-Speicherprofil, statistische
Verblindung und Konfidenzintervalle der subjektiven Bewertungen.
Diese Funktionen dürfen nicht stillschweigend vorgetäuscht werden.

## Von Null bis zur ersten Messung

Nehmen wir beispielsweise den nach
\`scripts/prepare_public_domain_films.py\` erstellten Ausschnitt
\`generated/caligari/portrait_probe/original_native.mkv\`
(Leerzeichen nur zur Veranschaulichung; tatsächliche Verzeichnis-ID
ist \`caligari\`):

\`\`\`bash
# Filmclip mit film-, szenen- und seedgebundener Unschärfe erzeugen.
python scripts/restoration_evaluation.py synthesize \
  --source benchmarks/public_domain_films/generated/caligari/portrait_probe/original_native.mkv \
  --film caligari --scene portrait_probe \
  --kind gaussian_blur --preset gentle --seed 2026 \
  --output-dir benchmarks/evaluation/caligari_portrait_blur_gentle

# Den exakt künstlich beschädigten Film (NICHT das Original) restaurieren.
python scripts/detail_enhance_video.py \
  --source benchmarks/evaluation/caligari_portrait_blur_gentle/degraded.mkv \
  --output benchmarks/evaluation/results/cas_caligari_portrait.mkv \
  --method cas --preset gentle --mix 100 \
  --intermediate-profile lossless

# Die vom Algorithmus erzeugte .mkv.sig.json prüft die Herkunft;
# das Ergebnis wird gegen die unveränderte Referenz gemessen.
python scripts/restoration_evaluation.py evaluate \
  --manifest benchmarks/evaluation/caligari_portrait_blur_gentle/manifest.json \
  --variant CAS=benchmarks/evaluation/results/cas_caligari_portrait.mkv \
  --output-dir benchmarks/evaluation/caligari_portrait_scored

# Erst nach Tests von mehreren unabhängigen Filmen aggregieren:
python scripts/restoration_evaluation.py summarize \
  --evaluation benchmarks/evaluation/caligari_portrait_scored/evaluation.json \
  --evaluation benchmarks/evaluation/melies_scored/evaluation.json \
  --evaluation benchmarks/evaluation/gertie_scored/evaluation.json \
  --bootstrap 1000 --seed 2026 \
  --output benchmarks/evaluation/film_level_summary.json
\`\`\`

Der neue Evaluator **führt die Kandidatenmodelle nicht automatisch
aus**. Sie sind weiterhin mit ihren bestehenden Backendskripten
und an identischen vorbereiteten Clips auszuführen. Das trennt
Modellgewichte/-rechte von Metrikcode und vermeidet einen
unbeabsichtigten CUDA-Test oder das Herunterladen nicht
freigegebener Gewichte.

## Fehler- und Datenleckkontrollen

- Referenz/Degradation werden per SHA-256 aus dem synthetischen
  \`manifest.json\` validiert.
- Jede Variante muss exakt dieselbe Bildzahl, FPS und Pixelgeometrie
  haben. Eine ffmpeg-Probe zählt tatsächlich dekodierte Frames.
- Varianten benötigen standardmäßig eine **passende Quell-SHA-256**
  über das ARP-Signatur-Sidecar oder ein Research-Provenance-Sidecar.
  Das \`--allow-unverified\`-Flag ist ein bewusst markierter
  Ausnahmefall; es **beweist keine Frame-Ausrichtung**.
- Die Referenz wird als 8-Bit-Video auf **demselben
  verlustfreien Encoderpfad** erzeugt wie die Beschädigung.
- Szene und Film sind die statistischen Einheiten; 24
  zusammengehörige Frames zählen **nicht** als 24 unabhängige Filme.
- Für Publikations- oder Release-Entscheidungen müssen
  Experimentkonfiguration und Modellversion vor der Auswertung
  fixiert werden. Filmbasierte Trennung von Entwicklung und Test
  verhindert inhaltliches Leaking über verschiedene Szenen
  desselben Films.
- Es gibt **keinen einheitlichen algorithmischen
  Qualitätssieger** und keine automatisierte Freigabe
  historisch korrekter Restaurierung.

## Beispiele für Fehlerinterpretation

Ein besserer PSNR nach Unsharp bei künstlichem Gauß-Blur
bedeutet **nur**, dass die Dekodierung dem gewählten sauberen
digitalen Referenzbild ähnlicher geworden ist.
Das beweist weder echte Auflösungsgewinnung noch
Originaltreue von Haaren/Gesichtern im historischen Film.

Ein besserer PSNR auf \`read_noise\` bedeutet **nicht**,
dass Filmkorn unbedenklich entfernt werden kann. Das
synthetische Rauschen stammt nicht von einem historischen
photochemischen Filmträger.

Eine negative Verschlechterung ist ebenso ein gültiges
wissenschaftliches Messergebnis und darf nicht versteckt werden.

## Technische und rechtliche Sicherheit

Für diese erste CPU-Phase: \`numpy\`, \`opencv-python-headless\`,
\`scikit-image\`, FFmpeg. Deren konkrete Versionsstände und
Lizenzbedingungen gehören in das Versuchsmanifest.
LPIPS-Gewichte, DISTS-Gewichte und generative Restaurierungsmodelle
werden **nicht** implizit heruntergeladen.
Die rechtliche Einordnung der historischen Filmwerke,
Digitalisierungen, neueren Restaurierungen und Filmtonspuren
bleibt unabhängig von der gemessenen Genauigkeit.
