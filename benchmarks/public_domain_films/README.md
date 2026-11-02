# ARP: mehrere gemeinfreie Stummfilme als Restaurierungs-Benchmark

**Forschungsstand: November 2026. Keine generelle Freigabe für kommerzielle Wiederverwendung.** Die Originalfilme sind nach den Wikimedia-Commons-Angaben gemeinfrei (USA und/oder Leben+70), aber **Digitalisierung, Restaurierung, eingefügte Musik oder Bearbeitungen können gesonderten Schutz oder Nutzungsbedingungen unterliegen**. Alle erzeugten Ausschnitte werden **ohne Ton** gespeichert. Die konkreten Transferrechte bleiben einzeln zu klären.

## Was im Repository liegt

- `benchmarks/public_domain_films/catalog.json` – sieben **zusätzliche** Filmtitel mit konkreten Commons-Quelldateien, vorgesehenen Zeitmarken, Testzielen, Rechte-/Transferwarnungen und Aufnahmeformat.
- `scripts/prepare_public_domain_films.py` – Download einer *konkreten* Commons-Version mit erfassten Checksums und anschließend Erstellung der Testclips.
- `tests/test_prepare_public_domain_films.py` – ohne Internet/GPU ausführbare Tests (inkl. tatsächlicher FFmpeg-Videos).
- `.github/workflows/public-domain-corpus.yml` – erzeugt pro Film separat einen herunterladbaren GitHub-Actions-Artefaktordner. Es werden **keine Filmdaten** in Git eingecheckt.
- `docs/step3-gpu-benchmark-protocol.md` – gemeinsames Protokoll für spätere gleich ausgerichtete CPU-/GPU-Vergleiche.

Das bisherige *Neighbors*-Testmaterial bleibt als achte, gesondert dokumentierte Quelle bestehen. Seine US-Public-Domain-Kennzeichnung ist **nicht automatisch** eine rechtliche Freigabe für Deutschland.

## Kuratierte Variation der Motive

| Film (Jahr) | Testszenen / Potenzial | Historischer Schwerpunkt | Quellenqualität / Rechtehinweis |
|---|---|---|---|
| [L'Arrivée d'un train en gare de La Ciotat (1896)](https://commons.wikimedia.org/wiki/File:L%27Arriv%C3%A9e_d%27un_train_en_gare_de_La_Ciotat_Completo.webm) | Annähernde Kamera-Bewegung, Menschen, Rauch | Sehr frühe dokumentarische Aufnahme | 540×360 YouTube-Digitalisierung |
| [The Great Train Robbery (1903)](https://commons.wikimedia.org/wiki/File:The_Great_Train_Robbery_%281903%29.webm) | Bewegungs-/Kontrasttests | Mehrere Darsteller und Szenenwechsel | Library-of-Congress-Ausgangsmaterial; Commons-360p/480p-Proxy |
| [Le Voyage dans la Lune (1902)](https://commons.wikimedia.org/wiki/File:Le_Voyage_dans_la_Lune_%281902%29.webm) | Kulissen, feine Strukturen, Trickeffekte | Méliès, frühe visuelle Effekte | Commons-360p/480p-Transcode, nicht der originale AV1-Upload |
| [Das Cabinet des Dr. Caligari (1920)](https://commons.wikimedia.org/wiki/File:The_Cabinet_of_Dr._Caligari.webm) | Kanten, schräge Bühnenarchitektur, Gesichter | Deutscher Expressionismus | Frühe Archive.org-Kopie; mögliches ungewöhnliches Bildformat |
| [Gertie the Dinosaur (1914)](https://commons.wikimedia.org/wiki/File:Winsor_McCay_%281914%29Gertie_the_Dinosaur.webm) | Handgezeichnete Linien, Weißflächen, Flackern | Zeichentrick | 2004-DVD-Transfer! **Zusätzliche Rechteprüfung** |
| [A Corner in Wheat (1909)](https://commons.wikimedia.org/wiki/File:Corner_in_Wheat.webm) | Innen-/Außenräume, Stoff, Flächen und Natur | Frühe US-Dramaturgie | 480×360 YouTube-Kopie |
| [Nosferatu (1922)](https://commons.wikimedia.org/wiki/File:Nosferatu_%281922%29.webm) | Dunkle Flächen, Schatten, mögliche Tönung | Deutscher Horror-Stummfilm | 2006-Restaurierung! **Zusätzliche Rechteprüfung** |

**Zeitmarken sind zunächst Proben:** Die zwölf plus zwei ausgewählten Fenster sind keine nach einer Sichtung bestätigten Film-Einstellungen. Es muss nach dem ersten Kontaktbogen kontrolliert werden, ob der Ausschnitt tatsächlich das angenommene Motiv, einen Schnitt, Vorspann oder einen Zwischentitel zeigt. Keine nicht überprüften Szenenbezeichnungen als Fakten behandeln.

## Pro Szene gespeicherte Testvarianten

```text
benchmarks/public_domain_films/generated/  # ignoriert durch .gitignore
  <film-id>/
    corpus_manifest.json              # Quelle, Commons SHA-1 + Download-SHA-256,
                                      # Zeitmarken, Dateihashes, Bildfrequenz, Abmessungen
    <scene-name>/
      original_native.mkv              # FFV1, gleiche Pixelmaße wie der konkrete Download
      synthetic_blur_mild.mkv          # künstlicher Gauß-Blur σ 0,7 px
      synthetic_blur_medium.mkv        # künstlicher Gauß-Blur σ 1,3 px
      unsharp_gentle.mkv               # FFmpeg Unsharp
      cas_gentle.mkv                   # FFmpeg CAS
      lanczos_reference.mkv           # nur wenn 128er-Maße ohne Crop/Stretch erreichbar
      diagnostics/
        denoising_comparison.json
        denoising_comparison.jpg
```

Die **synthetischen** Unschärfen sind eine *kontrollierte Benchmark-Degradation mit bekanntem Ausgangsbild*. Sie repräsentieren weder den tatsächlichen historischen Defekt noch einen originalen Restaurierungszustand. Das Label muss bei jeder späteren Auswertung bestehen bleiben.

Die Breiten/Höhen variieren **bewusst** entsprechend der Quelldigitalisierung; ein 540×360-Lumière-Transfer wird nicht als original 960×720 ausgegeben. Bilder verschiedener Filme oder Quellen-Auflösungen dürfen nicht direkt mit ihren pixelbasierten Laplacian-Werten bewertet werden. Wenn ein 128-divisibles VSR-Zielformat nur durch Zuschneiden oder Verzerren erreichbar wäre, bleibt die Lanczos-Vergleichsdatei **aus**, nicht automatisch korrigiert.

## Lokal erstellen

```bash
python scripts/prepare_public_domain_films.py \
  --film lumiere_train \
  --output-dir benchmarks/public_domain_films/generated/lumiere_train
```

Die Anwendung ruft die Wikimedia-Commons-Videoinformationen ab, prüft die ermittelte Original-SHA-1, lädt nur den explizit gewählten Original-Download oder Commons-VP9-360p/480p-Transcode, berechnet einen **lokalen SHA-256**, erstellt anschließend die ausgewählten Clips und löscht den heruntergeladenen vollständigen Film aus dem temporären Arbeitsverzeichnis. Bei nicht gefundenen Transcodes und nicht verfügbaren Medien wird abgebrochen, statt einen anderen Film oder unbemerkt den möglicherweise gigabytegroßen Master zu nehmen.

```bash
# Nur Commons-Metadaten überprüfen, ohne Mediendownload
python scripts/prepare_public_domain_films.py \
  --film melies_moon --output-dir benchmarks/public_domain_films/generated/melies_moon \
  --metadata-only

# Eigene Datei für dieselben zwei Clip-Zeitmarken nutzen.
# Nicht automatisch mit dem Commons-Transfer gleichsetzen.
python scripts/prepare_public_domain_films.py \
  --film great_train_robbery \
  --source-file input/LoC_Train_Robbery_own_transfer.mkv \
  --output-dir benchmarks/public_domain_films/generated/great_train_robbery
```

Voraussetzungen: FFmpeg/FFprobe und eine Python-Umgebung mit `numpy`, `opencv-python-headless` und den ARP-Skripten. Der Download ist auf **250 MiB pro Film** begrenzt; eine Quellenvariante oberhalb dieser Grenze muss gezielt geprüft und anders bearbeitet werden, nicht einfach ohne Schranke geladen werden. Beim ersten Lauf wird der Commons-Original-SHA-1 gespeichert; für vollständig unveränderliche Downloads sollten die ermittelten Werte anschließend in `catalog.json` festgeschrieben werden.

## Technische/ethische Grenzen

- Commons „gemeinfrei“ ≠ automatisch freie Rechte an **jeder** restaurierten Kopie, neuer Musik oder nachträglich hinzugefügten Anteilen. Besonders *Gertie* (DVD-Ausgabe 2004) und *Nosferatu* (Restaurierung 2006) müssen vor kommerzieller Weitergabe gesondert geprüft werden.
- Keine künstliche Verbesserung als „wiederentdeckte echte historische Details“ interpretieren; klassische Schärfung kann Kratzer und Filmkorn verstärken.
- GitHub-Artefakte sind **zeitlich befristet**; lokal gespeicherte Clips/Manifestdaten bleiben reproduzierbar über das Skript und überprüfbare Dateihashes.
- Bei Wikimedia-Commons-360p/480p-Proxys ist die tatsächliche Transferqualität **niedriger** als die des hochaufgelösten Masters. Für finale Qualitätsentscheidung zunächst einen zusätzlichen Benchmark mit geprüften Master-Kopien durchführen.
