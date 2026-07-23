# GAIA v2

Applicazione standalone (PyTorch + Streamlit) per la stima di **Pressione (kbar)**
e **Temperatura (°C)** dei sistemi di alimentazione vulcanica a partire dalla
composizione chimica del clinopirosseno (geotermobarometria).

Questa è una riscrittura self-contained della precedente app GAIA (vedi
`GAIA_legacy/` nel workspace di sviluppo originale), con un nuovo layer di
inferenza basato sui modelli PyTorch definitivi (luglio 2026), al posto dei
vecchi modelli TensorFlow.

## Cosa fa

1. L'utente carica un file (csv/xls/xlsx) con le analisi chimiche di
   clinopirosseno (ossidi maggiori in wt%).
2. L'app calcola le 11 componenti strutturali del clinopirosseno (stesso
   procedimento chimico della app legacy).
3. Un ensemble di reti neurali PyTorch (100 modelli per la pressione, 20 per
   la temperatura) predice P e T per ciascun campione, riportando anche
   l'incertezza (deviazione standard tra i modelli dell'ensemble).
4. I risultati possono essere scaricati in formato Excel.

## Cromo misurato vs non misurato

Esistono **due famiglie di modelli**, addestrate separatamente:

- **`with_chromium`**: usata quando il Cr2O3 è stato analizzato. Le componenti
  cromo-dipendenti (`CaCrTs`, `NaCrSi2O6`) vengono calcolate normalmente.
- **`without_chromium`**: usata quando il Cromo NON è stato misurato. **I
  modelli di questa famiglia sono stati addestrati impostando a zero le
  componenti `CaCrTs` e `NaCrSi2O6`** (dopo il preprocessing chimico, prima
  dell'addestramento/inferenza). Per essere coerenti con l'addestramento,
  l'app applica esattamente la stessa procedura di azzeramento prima di
  invocare questa famiglia di modelli.

Entrambi i flussi condividono la **stessa identica pipeline di preprocessing**
chimico (`gaia/preprocessing.py`); l'unica differenza è l'azzeramento delle
due componenti cromo-dipendenti e la scelta della famiglia di checkpoint.

La modalità va **sempre selezionata esplicitamente** dall'utente nell'interfaccia
(non viene mai dedotta automaticamente dai dati).

## Formato di input richiesto

Colonne richieste, in quest'ordine:

```
Index, sample, notes, notes.1, SiO2, TiO2, Al2O3, Cr2O3, FeO tot, MnO, NiO, MgO, CaO, Na2O, K2O, tot
```

- `Index`, `sample`, `notes`, `notes.1`: identificano il campione (testo libero).
- Ossidi maggiori in wt%: se un ossido non è stato analizzato o è sotto il
  limite di rilevabilità, inserire `0` o lasciare la cella vuota.
- `tot`: totale analitico, colonna informativa (non usata nel calcolo).

Un file di esempio è disponibile in `examples/input_example.xlsx` (derivato
dal template della app legacy) e un template vuoto in
`examples/input_template_empty.xlsx`.

## Installazione

```powershell
conda create -n GAIAv2_env python=3.9
conda activate GAIAv2_env
pip install -r requirements.txt
```

## Avvio locale

```powershell
cd GAIA_v2
streamlit run app.py
```

## Test

```powershell
cd GAIA_v2
python -m pytest tests -v
```

Tutti i test (23) sono eseguibili da dentro `GAIA_v2` senza alcuna dipendenza
da file o import esterni alla cartella. Includono:

- preprocessing chimico condiviso e deterministico;
- corretto azzeramento delle sole componenti cromo-dipendenti;
- validazione dell'input;
- caricamento dei checkpoint e modalità `eval()`;
- integrazione end-to-end per entrambi i flussi;
- **equivalenza numerica**: l'output del modello di inferenza semplificato
  (`gaia/models.py`, solo `net` + `regressor`) è confrontato contro una
  ricostruzione indipendente dell'architettura originale completa
  (con il ramo `discriminator`, mai usato in inferenza) caricata con gli
  stessi pesi. Tolleranza usata: `rtol=1e-5, atol=1e-6` (i pesi e la parte
  di rete rilevante sono identici bit per bit; la tolleranza copre solo
  eventuale non-determinismo interno di BLAS/PyTorch).

## Organizzazione dei modelli (artifacts/)

```
artifacts/
├── with_chromium/
│   ├── pressure/       # 100 checkpoint (mod_0_.pth ... mod_99_.pth)
│   └── temperature/    # 20 checkpoint (mod_0_.pth ... mod_19_.pth)
└── without_chromium/
    ├── pressure/       # 100 checkpoint
    └── temperature/    # 20 checkpoint
```

Ogni file `.pth` è un `state_dict` PyTorch puro (nessun modello serializzato
completo, nessuno scaler incorporato). Dimensione totale copiata:
**~486.7 MB (240 file)**. Essendo superiore alla soglia consigliata per un
repository Git "normale" (~100 MB), **si raccomanda l'uso di Git LFS** per
tracciare la cartella `artifacts/` se questo progetto verrà versionato in un
repository Git. Git LFS non è stato configurato automaticamente.

Non sono stati copiati: `global_history.pkl` (storico di training) e i file
`seq_sorted_bootstrap*.pickle` (indici di bootstrap), non necessari per
l'inferenza.

## Architettura dei modelli

Rete MLP con adattamento di dominio (solo il ramo `net` + `regressor` è usato
in inferenza; il ramo `discriminator`, usato in training per l'adversarial
domain adaptation, è ignorato):

- Pressione: `Linear(11→100) → ReLU → BatchNorm1d → Dropout(0.1) → Linear(100→100) → ReLU → Linear(100→100) → ReLU → Linear(100→1)`
- Temperatura: stessa struttura con larghezza 1000 invece di 100.

## Output e unità

- **Pressione**: kbar (media e deviazione standard dell'ensemble).
- **Temperatura**: °C (media e deviazione standard dell'ensemble).
- **`cpx_selection`**: flag booleano di qualità chimica (10 controlli:
  bilancio Wo/En/Fs, somma dei componenti, sito T, sito M, bilancio di
  carica, ecc. - stessa logica della app legacy). I campioni che non
  superano il controllo hanno predizioni impostate a `NaN` invece di un
  valore fittizio.

## Assunzioni scientifiche importanti

- Nessuno scaler di input (StandardScaler/MinMax) è applicato: le 11
  componenti derivano da un bilancio chimico che le vincola già in un range
  comparabile. Questo riflette esattamente la pipeline di training originale.
- La de-normalizzazione dell'output è una semplice moltiplicazione lineare
  fissa (`×10` per la pressione, `×1400` per la temperatura), non uno scaler
  fittato sui dati.
- Le componenti cromo-dipendenti azzerate sono esattamente `CaCrTs` e
  `NaCrSi2O6` (e nessun'altra), applicate dopo il preprocessing chimico e
  prima dell'inferenza.

## Limiti attuali

- L'app non ricalcola/valida automaticamente se il file caricato è coerente
  con la modalità Cromo scelta (es. tutte le colonne Cr2O3 a zero mentre è
  stato selezionato "Cromo misurato"): la scelta resta responsabilità
  dell'utente, per design (vedi requisiti).
- Nessuna gestione di autenticazione, database o deployment multi-tenant:
  l'app è pensata per un uso singolo-utente/locale o per deployment tramite
  Streamlit Community Cloud / server interno.
- I modelli sono stati validati sui dataset descritti nel workspace di
  sviluppo originale (GAIA1/GAIA2/Wieser); l'accuratezza su composizioni
  chimiche molto diverse da quelle di training non è garantita.

## Deployment su Streamlit Community Cloud

1. Versionare questa cartella (`GAIA_v2/`) in un repository Git dedicato
   (eventualmente con Git LFS per `artifacts/`, vedi sopra).
2. Su [share.streamlit.io](https://share.streamlit.io), collegare il
   repository e impostare `app.py` come entry point.
3. Assicurarsi che `requirements.txt` sia nella root del repository
   (già presente qui).

## Licenza e attribuzione

Basato sul progetto GAIA originale (Dipartimento di Fisica e Astronomia e
Dipartimento di Scienze della Terra, Università di Firenze). Nessuna licenza
esplicita era indicata nella app legacy (`GAIA_legacy/GAIA-main/README.md`
riporta solo "[In publ.]" come riferimento alla pubblicazione scientifica).
Si raccomanda di verificare/aggiungere una licenza esplicita prima di una
distribuzione pubblica.

