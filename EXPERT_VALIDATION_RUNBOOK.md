# Ekspertska validacija: operativni protokol v1

## Šta je sada spremno

Ovaj paket pokreće obavezni ekspertski pilot-gate za Es3 lokalizaciju. On ne
predstavlja eksternu validaciju ML regresora. Razvojni dataset, QC-DTW i pet ML
seedova zamrznuti su u `annotations/expert_validation_freeze_v1.json`; do kraja
pregleda i adjudikacije ne menjati foldove, QC, pragove, izbor kandidata ni model.

Pregled ima dva odvojena skupa:

- `core`: 17 stavki sa RGB videom uzorka i reference, od prvobitno izabranih 20;
  samo ovih 17 stavki ulazi u meru slaganja glavnog RGB pregleda;
- `supplemental`: tri dodatne stavke. Sa nerešenom stavkom `P_ID12_Es3`, koja je
  već u `core`, time su sva četiri nerešena pilot-slučaja pokrivena tačno jednom.

Ukupno svaki ekspert ocenjuje 20 jedinstvenih stavki. Izostavljeni su
`E_ID12_Es3`, `NE_ID20_Es3` i `P_ID4_Es3`, jer njihovoj referenci `NE_ID9_Es3`
nedostaje RGB. Posle tog izostavljanja kohorte i rangovi više nisu potpuno
balansirani. Pokrivenost glavnog izbora je 17/20; slaganje se računa na 17
pregledanih stavki i ne predstavlja rezultat za svih 20 prvobitnih kandidata.

## Pre dolaska eksperata

Pokrenuti proveru:

```powershell
.\.venv\Scripts\python.exe .\audits\verify_expert_validation_readiness.py
```

Mora pisati `"pilot_gate_ready": true`. Polje
`"strict_external_validation_ready": false` je očekivano: ovo je pilot-gate,
ne novi eksterni holdout.

Organizator dodeljuje šifre `expert_01` i `expert_02` i odvojeno čuva mapu šifra
– osoba. Eksperti ne smeju videti primarne pilot-oznake, klinički TS, predikciju
modela, DTW magnitudu ni međusobne odgovore pre zaključavanja oba pregleda.

## Rad eksperta

Za glavni krug:

```powershell
.\start_expert_validation.ps1 -Round core
```

Otvoriti `http://127.0.0.1:5050`, uneti dodeljenu šifru i oceniti svih 17
stavki. Prvo se prikazuje referentni RGB video, zatim referenca i video uzorka
jedan pored drugog. RGB je glavni dokaz; prikaz frejmova je sklopljen i dostupan
kao pomoć. Ocenjuje se samo imenovana komponenta u označenom intervalu, a ne ukupan
klinički kvalitet izvođenja. Značenje polja je u `ANNOTATION_GUIDE.md`.

- `correct`: u komponenti/intervalu nema vidljive greške;
- `error`: postoji vidljiva greška; obavezno izabrati tip i težinu;
- `uncertain`: pokret je vidljiv, ali odluka nije pouzdana; obavezna beleška;
- `ungradable`: vidljivost, tracking ili dokaz nisu dovoljni; obavezna beleška.

Tek kada su sve stavke popunjene, ekspert zaključava pregled. Collect-only režim
ni tada ne prikazuje primarne oznake ili slaganje. Posle izvoza
`second_review_completed.csv` ekspert završava sesiju. Drugi ekspert koristi drugi
ID i radi isti postupak nezavisno. Preporučeni nazivi arhivskih izvoza su
`core_expert_01.csv` i `core_expert_02.csv`.

Zatim svaki ekspert radi dodatne tri stavke:

```powershell
.\start_expert_validation.ps1 -Round supplemental
```

Te stavke se analiziraju i adjudiciraju zasebno od glavnih 17 stavki.
Pre pokretanja sledećeg kruga na istom portu zaustaviti prethodni sa Ctrl+C.
Alternativno koristiti `-Port 5051` i otvoriti odgovarajući port u pregledaču.

## Tumačenje video prikaza

Oba podrazumevana kruga koriste RGB i collect-only režim. Stari nazivi opcija
`core-video-preview` i `supplemental-video-preview` ostaju aliasi za iste krugove.
U izvozu ostaje tehnička oznaka `preview_not_validation`: ona označava da nije
potvrđen strogi holdout i sinhronizacija; RGB je ipak glavni dokaz ovog ekspertskog
pilota. Interval 0–100% znači ceo snimak. Zajednički klizač ne garantuje podudaranje
faza pokreta. Samo razlika položaja na klizaču nije dokaz greške u tajmingu.
Za nedovoljan dokaz koristiti `ungradable`, uz obrazloženje.

## Posle oba pregleda

Ne počinjati adjudikaciju dok oba eksperta ne zaključaju i izvezu oba skupa.
Prvo se računaju tačno slaganje, Cohenova kappa, binarno slaganje i pokrivenost
na `core` skupu. Svako neslaganje u oznaci, tipu ili težini, kao i svaki
`uncertain`/`ungradable`, ide trećem ekspertu ili dokumentovanom konsenzusu.
Sačuvati oba originalna izvoza, finalni adjudicirani CSV, agreement JSON, bazu i
zamrznuti `.round` direktorijum; ništa od toga ne prepisivati.

Tek posle ovog gate-a protokol se zaključava za novi held-out skup. Za tvrdnju o
strogoj eksternoj validaciji i dalje su potrebni novi subjekti, potvrđene granice
pokreta i sinhronizacija RGB/skeleta, kao i kompletan inventar treninga modela.
