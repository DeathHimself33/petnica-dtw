# Audit projekta — nalazi i plan popravki

Datum: 2026-09-05. Pregledano lokalno stanje sa HEAD `4050628` i postojećim
nekomitovanim dijagnostičkim izveštajima. Ovo je tehnički audit i razvojna analiza,
ne klinička potvrda. Produkcioni kod, sirovi podaci, ocene, modeli, postojeći
rezultati i stvarna ekspert baza nisu menjani.

**Glavni nalaz:** postoje konkretne greške u upravljanju ML eksperimentima i
ekspertnim ocenama, kao i nedostaci reprezentacije i porekla podataka. Provereni
postojeći rezultati ipak nisu posledica pogrešnog računanja metrika ili zamene
ulaznih podataka. Es4 zahteva istraživanje signala koje model uopšte dobija;
samo jači model ili opšte „čišćenje” nemaju dokazanu korist.

## Šta je provereno i šta je prošlo

| Provera | Rezultat |
|---|---|
| Postojeći unittest skup | 89 testova prošlo |
| Nezavisni DTW oracle, iscrpno nabrajanje putanja na malim nizovima | 150 kvadratnih + 150 angularnih slučajeva prošlo |
| Ponovno generisanje manifesta iz izvora | Svih 390 redova se poklapa nakon standardne CSV serijalizacije praznih vrednosti |
| Inventar svih JointPosition fajlova | 402 fajla: 400 u Es1–Es5, dva dodatna u Es6 |
| Ponovno generisanje ML tenzora iz sirovih podataka | Svih 12 sačuvanih nizova identično; 338 uzoraka, 52 isključenja |
| Normalizacija u 25 ML checkpoint-a | Poklapa se sa ponovnim fitovanjem samo na odgovarajućim trening foldovima |
| Razdvojenost ML train/validation/test subjekata | Prošla za postojeći petofoldni dataset |
| Ponovna inferencija svih 25 checkpoint-a na originalnom GPU-u | Maksimalna razlika od sačuvanih predikcija: **0,0** |
| Postojeći DTW izveštaji | 13 skupova: mete, MAE/RMSE/Pearson/Spearman i hash izvora se poklapaju |
| ML i QC-DTW poređenje | Ista 338 sample ID-a i isti test foldovi |
| Ensemble | Proseci, standardne devijacije između seed-ova i apsolutne greške se poklapaju |

CPU inferencija je odstupala do 0,0201 TS od GPU predikcija; ponavljanje na
originalnom GPU-u uklonilo je razliku potpuno. To nije dokaz neispravnih
checkpoint-a. Nijedan model nije ponovo treniran.

Pregled obuhvata loader i manifest audit, preprocessing i njegove dijagnostike,
oba DTW jezgra, reference i kalibraciju, grupisanje subjekata, statistiku,
QC i lokalizaciju, izvoz ML podataka, model i trening, agregaciju seed-ova,
pilot oznake, ekspert aplikaciju i njene šablone, CLI i dokumentaciju. Postojeći
testovi ne pokrivaju potvrđene scenarije resume/provenance i više recenzenata;
nezavisni reproduktori su zato odvojeni u `audits/`.

## Potvrđene greške i nedostaci ugovora

P1 znači rešiti pre sledećeg relevantnog eksperimenta ili ekspert runde;
P2 znači planirana popravka pre širenja projekta; P3 znači manji nedostatak.
Prioritet nije tvrdnja da su postojeći rezultati nužno već oštećeni.

### A01 — P1: resume može starim predikcijama pripisati novi eksperiment

Mesto: `train_ml_baseline.py:505`, naročito grana na liniji 510;
`analyze_ml_results.py:110`.

Kopirano je samo pet starih `predictions.csv` u privremeni direktorijum, pa je
pozvan `--resume --seed 999123 --channels 32`. Program je zadržao svih 338 starih
predikcija, zapisao novi seed i konfiguraciju, a agregator ih prihvatio kao
kompletan rezultat. Checkpoint-i nisu bili potrebni.

**Posledica:** budući test robustnosti može izgledati kao novi seed ili model
iako se ništa nije promenilo. Izvorna NPZ putanja/hash i potpuna konfiguracija
nisu vezani za fold artefakte. Agregator ne proverava jednaku konfiguraciju
između run-ova osim identiteta uzoraka i deklarisanih seed-ova.

**Popravka:** nepromenljiv run manifest sa hash-em NPZ-a, foldova, koda i
konfiguracije; atomarno zapisivanje završenog folda; resume samo uz potpuno
poklapanje. Agregator mora proveriti poreklo i usaglašenost konfiguracija.
Dodati regresioni test koji pokušava promenu seeda, modela i podataka.

### A02 — P2: exporter dopušta šest foldova, trainer izvršava samo pet

Mesto: `src/kimore_ml_data.py:198`, `train_ml_baseline.py:222` i `:505`.

Stvarno je napravljen izvoz sa šest foldova. Orkestracija treninga proverena je
sa testnom zamenom funkcije treninga: prijavila je završene foldove 1–5 i
280 OOF uzoraka; **58 uzoraka iz folda 6 nikada nije dobilo test predikciju**.
Skup i dalje ima 338 uzoraka. Nije izvršen trening.

**Popravka:** odmah odbiti sve osim pet foldova, ili uvesti zajednički eksplicitan
split protokol koji trainer i agregator poštuju. Na kraju obavezno proveriti
tačno jedan OOF red za svaki očekivani uzorak, a ne samo deklarisani broj foldova.
Postojeći petofoldni artefakti nisu pogođeni ovim reproduktorom.

### A03 — P1: adjudikacija jednog pregleda ulazi u drugi pregled

Mesto: `src/kimore_expert_review/model.py:245`, `:356`, `:394`.

Primarni ključ adjudikacije je samo `(sample_id, candidate_rank)`. U testnoj bazi
dva recenzenta su zaključala preglede. Odluka adjudikatora za jedan pregled
pojavila se kao završena adjudikacija i u izvozu drugog recenzenta. Naknadna
odluka može prepisati prethodnu.

**Popravka:** jasno izabrati da li je adjudikacija zajednički panel ili odluka za
konkretan par/krug. U oba slučaja trajno vezati odluku za identifikovane ulazne
ocene, rundu i verziju dokaza. Čuvati istoriju odluka umesto tihog prepisivanja.

### A04 — P1: zaključan pregled ne zamrzava primarne oznake i dokaz

Mesto: `src/kimore_expert_review/__init__.py:92`,
`src/kimore_expert_review/model.py:205` i funkcija `_cropped_evidence_cached`.

Baza proverava hash reda za ocenjivanje, ali ne primarne oznake ni slike dokaza.
Promena primarnog CSV-a i ponovno pokretanje nad istom testnom bazom promenili su
slaganje već zaključanog pregleda sa **0% na 100%**. Slike se takođe mogu zameniti;
mtime osvežava cache, ali ne obezbeđuje istoriju prikazanog dokaza.

**Popravka:** zamrznuti kompletan paket runde: queue, primarne oznake, slike ili
klipove, protokol, verziju aplikacije i hash-eve. Promena zahteva novu rundu.
Izvozi moraju nositi te identifikatore.

### A05 — P1: rang kandidata nije stabilan identitet intervala

Mesto: `src/kimore_apply_pilot_labels.py:81`;
`src/kimore_expert_review/model.py:52` i `:113`.

U kopiji queue-a promenjeni su komponenta i početni frejm, uz isti sample ID i
rang. `merge_labels` je bez prigovora prikačio staru oznaku novom intervalu.
Posle promene QC-a, reference ili lokalizacije rang lako može označavati drugi
pokret. App hash queue-a štiti deo toka, ali ne popravlja nezavisno spajanje
oznaka i ne proverava identitet primarne oznake do nivoa sadržaja intervala.

**Popravka:** stabilan candidate ID izveden iz verzije run-a, uzorka, reference,
komponente i granica intervala; striktna provera nepromenljivih polja pri merge-u.

### A06 — P1: finalna adjudikacija ne dozvoljava „nije moguće oceniti”

Mesto: `src/kimore_expert_review/model.py:183` i
`src/kimore_expert_review/templates/adjudication.html`.

Prvi i drugi pregled dopuštaju `ungradable`, a finalni formular i validator
prihvataju samo `correct`/`error`. Reproduktor potvrđuje odbijanje neocenjivog
slučaja. Loš dokaz ne postaje ocenjiv time što je prosleđen adjudikatoru.

**Popravka:** dozvoliti konačno izuzeće zbog neocenjivosti, sa obrazloženjem;
jasno definisati imenilac metrika i pokrivenost. Za `uncertain` predvideti
dodatni dokaz ili nerazrešen status, umesto prinudnog binarnog izbora.

### A07 — P1: podrazumevana runda ne obuhvata tri nerešene pilot stavke

Mesto: izbor drugog pregleda u `src/kimore_apply_pilot_labels.py` i
podrazumevani `second_review_queue.csv`; zahtev u `PILOT_LABEL_REPORT.md:85`.

Od četiri preliminarno nesigurne stavke, tri nisu među 20 stavki aplikacije:
`E_ID14_Es3` rang 2, `S_ID10_Es3` rang 1 i `S_ID8_Es3` rang 1.
Zato završetak podrazumevanog pregleda i njegovih adjudikacija ne ispunjava
zahtev da se razreše sve četiri nesigurne stavke.

**Popravka:** posebna eksplicitna lista svih nerešenih pilot slučajeva; zadržati
originalnih 20 stavki za unapred definisano merenje slaganja i odvojeno prikazati
dodatne slučajeve. Ne menjati zaključanu rundu naknadnim dodavanjem redova.

### A08 — P2: reviewer ID je šifra po dogovoru, ne autentifikacija

Mesto: `src/kimore_expert_review/__init__.py:241`.

Svež testni browser je uneo postojeći zaključani ID i dobio `/agreement` sa
statusom 200, bez tajne. Različit tekst za adjudicator ID ne dokazuje drugu
osobu. Ovo je ograničenje integriteta lokalnog radnog procesa, naročito ako se
aplikacija kasnije deli većem broju recenzenata.

**Popravka:** kontrolisana dodela reviewer tokena/naloga i uloga, uz vezivanje
ocene za dodeljenu osobu. Ako ostaje potpuno lokalni pilot, jasno dokumentovati
da nezavisnost obezbeđuje organizator, a ne aplikacija.

### A09 — P2: rezervni prikaz skeleta odseca donje delove nogu

Mesto: `src/kimore_pilot_review.py:136` i `:145`.

SpineBase se crta na y=470 u slici visine 540, uz skalu 360/body_height.
Za referencu `NE_ID9_Es3`, koja nema RGB, oba članka i stopala su izvan slike u
prvom frejmu. To je potvrđeno koordinatama i vizuelnim pregledom generisane slike.

**Popravka:** centrirati i skalirati prema kompletnom prikazanom skeletu sa
marginama, uz vremenski stabilnu skalu. Ponovo generisati i proveriti dokaz pre
nove ekspert runde; ne zamenjivati dokaz postojećoj zaključenoj rundi.

### A10 — P2: pilot podrazumevano čita staru lokaciju rezultata

Mesto: `src/kimore_pilot_review.py:426`, `run_experiment.py:105`,
`ANNOTATION_GUIDE.md:3`.

Novi pojedinačni run piše u `results/interpretable_dtw/Es3/`, dok pilot čita
`results/interpretable_dtw/annotation_queue.csv`. Na ovom računaru stari queue
postoji; na čistom okruženju sled podrazumevanih komandi nije usaglašen.

**Popravka:** eksplicitan run ID ili obavezni manifest ulazne runde; uskladiti
CLI i dokumentaciju. Postojeći pilot treba označiti kao zamrznut istorijski run.

### A11 — P3: broj stavki na stranici slaganja prikazuje Python metodu

Mesto: `src/kimore_expert_review/templates/agreement.html`, izraz `summary.items`.
Jinja bira metodu rečnika `items`, umesto vrednosti ključa. Testni HTML sadrži
`built-in method items`. JSON statistika ostaje ispravna.

**Popravka:** koristiti `summary['items']` i proveriti prikaz konkretnog broja.

## Izvorni podaci i moguća poboljšanja

### D01 — P1 za poreklo oznaka: interni ID nije proveravan

`src/kimore_dataset_audit.py:196` čita ocene, ali ne proverava interni Subject ID.
Nezavisna prethodna provera pokrila je 398 workbook kopija kod 78 subjekata.

- E_ID16/Es2 ima interni E_ID17, dok četiri druge E_ID16 kopije imaju ispravan
  ID i identične ocene. Metadata greška je moguća, ali nije automatski dokazana.
- Svih pet NE_ID2 kopija ima interni E_ID1; ocene se razlikuju od stvarnog E_ID1.
  Ovo ostaje nerešeno pitanje identiteta i odnosi se na svih pet vežbi.
- Pogrešno imenovane E_ID1 kopije kod E_ID3 postojeći loader već ignoriše.

Potrebna je evidencija porekla i odluka po izvoru; ne zamenjivati ocene nagađanjem.

### D02 — P2: četiri identična para mogu omogućiti povratak uzoraka

SHA-256 inventar pronašao je četiri grupe identičnih JointPosition fajlova:
E_ID14_Es4, E_ID15_Es4, NE_ID24_Es5 i P_ID3_Es5. `find_exactly_one` na
`src/kimore_dataset_audit.py:183` ih tretira kao nerešene višestruke snimke.

To nisu različiti subjekti sa dupliranim podacima, niti potvrda da svaki folder
sa više fajlova sadrži samo kopije. Za ova četiri slučaja pripremiti sidecar
izbor kanonskog position fajla i proveriti odgovarajući timestamp/orientation/RGB
paket. Potom proveriti QC; nije unapred garantovano da će sva četiri ući u model.
Sirove kopije ne brisati.

### D03 — P2, izvorna anomalija: dve Es5 ocene odstupaju od PO + CF

E_ID3_Es5: TS=44,6667; PO=15; CF=29. E_ID4_Es5: TS=43,3333;
PO=12,6667; CF=30. Razlika je +0,6667 u oba slučaja i ponavlja se u svih pet
odgovarajućih workbook kopija. Druge kompletne tabele prolaze ovu proveru.

Ovo nije greška našeg mapiranja kolona. Razjasniti definiciju/agregaciju kod
izvora pre korekcije; sačuvani TS ne prepisivati zbirom bez potvrde.

### D04 — poznata oštećenja i vremenske praznine

Četiri position fajla imaju red od 200 umesto 100 vrednosti: E_ID8_Es1,
P_ID3_Es1, P_ID5_Es1 i S_ID6_Es5. Loader ih ispravno odbija. Moguća popravka
spojenih redova zahteva zasebnu proveru vremenskog i orijentacionog toka.

Među uspešno parsiranim fajlovima nisu nađene nevažeće tracking vrednosti,
nefinite position koordinate, identični susedni position frejmovi ili
vremenske oznake koje ponavljaju ili smanjuju prethodnu vrednost. U uparenim uspešno pročitanim
tokovima nisu nađena neslaganja broja frejmova. Praznine u vremenu postoje u
svih pet vežbi; apsolutna jedinica vremena nije pretpostavljena.

| Vežba | Sirovih position fajlova | Sa bar jednim razmakom >2× medijane | >5× medijane |
|---|---:|---:|---:|
| Es1 | 81 | 59 | 7 |
| Es2 | 80 | 55 | 12 |
| Es3 | 77 | 50 | 8 |
| Es4 | 82 | 47 | 7 |
| Es5 | 80 | 52 | 7 |

Brojevi uključuju kopije i isključene snimke; nisu procenat evaluacione populacije.
Dva dodatna Es6 snimka (NE_ID1 i NE_ID2) su van definisanih pet zadataka i ne
treba ih automatski priključivati nekoj vežbi.

### D05 — pokrivenost nije jednaka među kohortama

338/390 kombinacija subjekat–vežba je u zajedničkoj ML/QC-DTW evaluaciji.
Isključenja obuhvataju i manifest i QC; zato se ne smeju sva pripisati QC-u.
Na Es3 ostaje 6/10 stroke snimaka, a na Es4 5/8 back-pain snimaka. Izveštavati
pokrivenost po vežbi i kohorti uz metrike; uspeh na zadržanim snimcima ne dokazuje
uspeh na isključenima. Detaljna tabela je u `evaluation_artifacts.json`.

## Metodološke rupe koje popravka softvera sama neće rešiti

1. **Reprezentacija gubi relevantne vrste pokreta.** Devet vektora uklanja
   translaciju celog tela. Sintetički pomeraj karlice ne menja vektore; globalni
   nagib oko odgovarajuće ose može ostati nevidljiv lokalnim kostima i body-forward
   vektoru. Priloženi Es4 MATLAB kod koristi X/Z putanju SpineBase, a Es2 koristi
   nagib trupa. To su motivacije za dodatne feature-e, ne dokaz uzroka slabog TS.
2. **QC nije potpuna provera fizičke verodostojnosti.** Posebna anatomska pravila
   postoje za Es3. Tracking state 2 nije garancija dobrog skeleta; prethodna Es2/4
   dijagnostika nalazi neke velike skokove i posle postojeće obrade. Potrebno je
   oceniti označene tranzicije, a ne proglasiti svaku brzinu greškom i sve izravnati.
3. **Vreme i broj ponavljanja nisu adekvatno obuhvaćeni.** DTW bez ograničenja može
   poravnati ponovljen ili izostavljen segment. ML uniformna mreža ima 128 tačaka
   po indeksu frejma, bez izvornih trajanja. Gubitak ovih informacija je dizajnersko
   ograničenje; porediti vremenske/repetition feature-e u kontrolisanoj ablaciji.
4. **Poređenje metoda zavisi od protokola.** Es3 postojeći plain/Yu imaju 76
   uzoraka, QC 68. Pojedinačni i zajednički all-exercise Es3 run nisu ista procena.
   Za zaključke koristiti isti skup i iste foldove; ML i all-exercise QC to već rade.
5. **Razvojna evaluacija je već viđena.** Više seed-ova meri varijaciju treninga
   na istim foldovima, ne nezavisnu eksternu validaciju. Fixed-OOF bootstrap ne
   obuhvata ponovno biranje reference, QC-a, arhitekture i kalibracije. Dokumentacija
   ovo uglavnom pravilno označava; očuvati tu ogradu u završnom radu.
6. **Lokalizacioni ground truth još ne postoji.** Pilot 40 error / 56 correct /
   4 uncertain jeste preliminarni neklinički prolaz. Slaganje čoveka sa tim
   prolazom nije slaganje dva klinička eksperta. Top-5 kandidati mogu dati procenu
   preciznosti među kandidatima; za recall treba označiti i propuštene greške u
   celom zasebnom snimku, uz definiciju komponente, granica i temporalnog preklapanja.
7. **Statične slike ograničavaju timing procenu.** Tri frejma intervala i pregled
   snimka nisu zamena za reprodukciju pokreta. Provera broja RGB frejmova nije
   dokaz tačne vremenske sinhronizacije. Pre timing oznaka proveriti klipove i
   poravnanje na konkretnim događajima.
8. **Prikazani shoulder angle je ugao ose.** `src/kimore_pilot_review.py:225`
   koristi apsolutni dot proizvod, pa su suprotne orijentacije ruke izjednačene.
   Precizno imenovati meru ili uvesti orijentisanu/joint definiciju pre stručne
   interpretacije; ne predstavljati je kao pun klinički opseg pokreta.

## Predlog redosleda rada

1. Zamrznuti trenutne eksperimente kao razvojni baseline; dodati nepromenljivo
   poreklo i ispraviti A01/A02 pre novih treninga.
2. Pre slanja ekspertima rešiti A03–A09 i identitet paketa; dovršiti protokol za
   neocenjive slučajeve, dodatnu adjudikaciju i poseban held-out skup.
3. Uvesti sidecar registar izvornih anomalija i identičnih kopija (D01–D03), sa
   razlozima i hash-evima. Napraviti verziju dataseta, bez prepisivanja originala.
4. Za Es2/Es4 odvojeno testirati: postojeći feature-i; dodatni nagib trupa;
   putanja karlice; vreme/ponavljanja; zatim validirane QC promene. Svaka ablacija
   sa istim splitovima, istom populacijom za direktno poređenje i pokrivenošću.
5. Tek potom proceniti korist promene modela. Koristiti više seed-ova i unapred
   definisati šta je poboljšanje; ne birati filtere po najlepšem OOF rezultatu.

## Artefakti, reprodukcija i granice

`audits/full_project_verification.py`: oracle, resume i app reproduktori,
pilot i ML tenzori/normalizacija. Zahteva projektni venv.

`audits/full_dataset_inventory.py`: ponovni manifest, svi position fajlovi,
parovi timestamps/orientations i hash duplikati; projektni venv.

`audits/check_checkpoint_inference.py`: ista GPU inferencija iz 25 checkpoint-a;
projektni CUDA venv. Učitava isključivo poznate lokalne projektne checkpoint-e.

`audits/check_evaluation_artifacts.py`: nezavisna provera metrika, hash-eva,
ensemble-a i pokrivenosti; numpy i pandas (korišćen bundled Python, pošto
projektni venv nema pandas). Pandas nije potreban produkcionom pipeline-u.

`audits/check_six_fold_contract.py`: stvarni šestofoldni izvoz uz mock trening.

Detalji su u `results/full_project_audit/`: `verification.json`,
`dataset_inventory.json`, `checkpoint_gpu_inference.json`,
`evaluation_artifacts.json`, `six_fold_contract.json`, `target_sum_sources.json`
i `fallback_NE_ID9_Es3.png`. Ovaj direktorijum je Git-ignored; za arhiviranje
istraživanja sačuvati verzionisan paket dokaza uz izveštaj.

Nisu ručno odgledani svi RGB snimci niti klinički potvrđene oznake; nije
izvršena nova kompletna obuka, hardverski test na drugom računaru ili nezavisna
eksterna evaluacija. DTW rekurzije su proverene oracle-om i sačuvani rezultati
aritmetički, ali sve velike DTW matrice nisu ponovo računane. Pokušaj pristupa
punom Yu–Xiong radu kroz MDPI/PMC nije dao dostupan pun tekst, pa jednakost sa
svakom jednačinom izvornog rada nije proglašena potvrđenom. To su eksplicitne
granice ovog audita, a ne prećutno zatvorene stavke.
