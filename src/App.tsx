import { useEffect, useEffectEvent, useRef, useState } from 'react'
import {
  Activity,
  AlertCircle,
  ArrowDownToLine,
  Brain,
  Check,
  ChevronRight,
  CircleHelp,
  Contrast,
  ClipboardCheck,
  Download,
  FileText,
  FileImage,
  FolderOpen,
  Gauge,
  HardDrive,
  ImagePlus,
  Layers3,
  LoaderCircle,
  Maximize2,
  Menu,
  Microscope,
  ScanLine,
  ShieldCheck,
  Sun,
  RotateCw,
  RotateCcw,
  Upload,
  X,
} from 'lucide-react'
import './App.css'

type Screen = 'review' | 'training' | 'evaluation' | 'dashboard'
type CaseRecord = { id: string; filename: string; created_at: string; status: string }
type ApiStatus = {
  api_ready: boolean
  model_ready: boolean
  ml_dependencies_ready: boolean
  model_name: string | null
  classes: string[]
  training: boolean
  training_message: string | null
}
type Metrics = {
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  classes: string[]
  confusion_matrix: number[][]
  per_class?: Record<string, { precision: number; recall: number; f1_score: number; support: number }>
  evaluation_split?: string
  evaluated_at: string
} | null
type AnalysisResult = {
  result_type: 'research_prediction'
  disclaimer: string
  predicted_class: string
  confidence: number
  probabilities: Record<string, number>
  tumor_present: boolean | null
  heatmap_data_url: string
  explanation_note: string
  uncertainty: { flagged: boolean; reason: string; method: string }
  model_provenance: { model_name: string | null; architecture?: string; trained_at?: string }
  quality_metrics: { score: number; contrast: number; sharpness: number; entropy: number; foreground_ratio: number; flags: string[]; interpretation: string }
  robustness_check: { agreement: number; stable: boolean; variants: { variant: string; predicted_class: string; confidence: number }[]; interpretation: string }
  segmentation?: { available: boolean; mask_data_url: string | null; volume_cm3: number | null; reason: string }
  severity_grade?: { value: string | null; status: string; reason: string }
}
type ScanMetadata = {
  format: string
  modality: string | null
  rows: number
  columns: number
  pixel_spacing_mm: number[] | null
  slice_count?: number
  slice_index?: number
  anonymized?: boolean
}
type DatasetStatus = {
  ready: boolean
  path: string
  preparing: boolean
  message: string | null
  error: string | null
  metadata: {
    image_count: number
    license: string
    citation: string
    split_counts: Record<string, Record<string, number>>
  } | null
}
type ResearchDataset = {
  id: string
  name: string
  publisher: string
  url: string
  access: string
  license: string
  image_count: number | null
  split_summary: string
  classes: string[]
  notes: string[]
}

const emptyStatus: ApiStatus = {
  api_ready: false,
  model_ready: false,
  ml_dependencies_ready: false,
  model_name: null,
  classes: [],
  training: false,
  training_message: null,
}

type ConditionGuidanceContent = {
  meaning: string
  cause: string
  nextSteps: string
  treatment: string
}

const conditionGuidance: Record<string, ConditionGuidanceContent> = {
  glioma: {
    meaning: 'Glioma is a broad group of tumors that begin in glial cells. This model cannot confirm a tumor or identify subtype, grade, molecular markers, size, or location.',
    cause: 'The cause of most adult brain tumors is not known. This single-image model cannot determine an individual cause or risk factor.',
    nextSteps: 'Ask a radiologist or treating clinician to review the complete MRI series and official report. They may decide whether additional imaging, neurological assessment, or tissue testing is appropriate.',
    treatment: 'Care depends on a confirmed tumor type, grade, molecular findings, location, symptoms, and overall health. Specialists may discuss monitoring, surgery, radiation, medicines, or a clinical trial. There is no single treatment or cure that applies to every glioma.',
  },
  meningioma: {
    meaning: 'Meningiomas arise from the membranes covering the brain or spinal cord. Some grow slowly; this model cannot confirm one or determine its grade, growth, or clinical significance.',
    cause: 'A cause cannot be identified from this image or model label. Many adult brain tumor causes are unknown; a clinician can review personal history and relevant risk factors.',
    nextSteps: 'Have the complete MRI and official report reviewed by a clinician. Follow-up may depend on the finding, its location, growth over time, and whether symptoms are present.',
    treatment: 'Depending on a confirmed diagnosis and clinical factors, care may range from active monitoring to surgery or radiation. The model cannot determine whether treatment is needed or which option is appropriate.',
  },
  no_tumor: {
    meaning: 'The model favored its “no tumor” category among four labels. This is not a radiology report and does not rule out a tumor, another abnormality, or a cause for symptoms.',
    cause: 'This category does not identify a cause or explain symptoms. The model only compares this image with its training labels.',
    nextSteps: 'Review the official radiology report and discuss ongoing concerns with a clinician, even when the model predicts “no tumor.” The model analyzes one 2D image, not a full MRI study.',
    treatment: 'No tumor-specific treatment conclusion follows from this prediction. A clinician should decide whether the scan, symptoms, or other tests need follow-up.',
  },
  pituitary: {
    meaning: 'This label points to the dataset’s pituitary category. Pituitary tumors vary; the model cannot confirm a tumor, specify its type, or tell whether hormones or nearby structures are affected.',
    cause: 'The model cannot identify a cause. Causes and relevant risk factors require clinical history and a confirmed diagnosis.',
    nextSteps: 'Discuss the official MRI report with a clinician. If a pituitary finding is confirmed, clinicians may consider hormone testing, vision assessment, or specialist review.',
    treatment: 'Depending on the confirmed type and effects, specialists may consider monitoring, medicines for selected hormone-producing tumors, surgery, or radiation. The model cannot select a treatment or determine a cure.',
  },
}

function ConditionGuidance({ predictedClass }: { predictedClass: string }) {
  const guidance = conditionGuidance[predictedClass] ?? conditionGuidance.no_tumor
  const adultCnsSource = 'https://www.cancer.gov/types/brain/patient/adult-brain-treatment-pdq'
  const pituitarySource = 'https://www.cancer.gov/types/pituitary/patient/pituitary-treatment-pdq'

  return (
    <section className="panel condition-guidance" aria-labelledby="condition-guidance-title">
      <div className="condition-guidance-heading">
        <div>
          <div className="eyebrow">EDUCATIONAL CONTEXT</div>
          <h2 id="condition-guidance-title">About the {predictedClass.replaceAll('_', ' ')} label</h2>
        </div>
        <span className="education-badge"><ShieldCheck size={14} /> General information</span>
      </div>
      <p className="condition-guidance-disclaimer">This is not a diagnosis. The displayed percentage is a model score, not an individual medical likelihood. The model can be wrong and cannot determine cause, severity, or cure from an image. Use the official radiology report and a qualified clinician for medical decisions.</p>
      <div className="condition-guidance-grid">
        <div><h3>What it can mean</h3><p>{guidance.meaning}</p></div>
        <div><h3>Cause</h3><p>{guidance.cause}</p></div>
        <div><h3>What to do next</h3><p>{guidance.nextSteps}</p></div>
        <div><h3>Treatment and cure</h3><p>{guidance.treatment}</p></div>
      </div>
      <div className="urgent-care-note"><AlertCircle size={16} /><p><strong>Seek emergency care now</strong> for a first seizure, sudden severe headache, new weakness or numbness, sudden speech or vision problems, confusion, or loss of consciousness.</p></div>
      <p className="condition-guidance-sources">Patient information: <a href={adultCnsSource} target="_blank" rel="noreferrer">National Cancer Institute: adult CNS tumors</a>{predictedClass === 'pituitary' && <> · <a href={pituitarySource} target="_blank" rel="noreferrer">NCI: pituitary tumors</a></>}</p>
    </section>
  )
}

function App() {
  const [screen, setScreen] = useState<Screen>('review')
  const [role, setRole] = useState<'admin' | 'radiologist' | 'patient'>('radiologist')
  const [status, setStatus] = useState<ApiStatus>(emptyStatus)
  const [dataset, setDataset] = useState<DatasetStatus>({ ready: false, path: '', preparing: false, message: null, error: null, metadata: null })
  const [researchDatasets, setResearchDatasets] = useState<ResearchDataset[]>([])
  const [cases, setCases] = useState<CaseRecord[]>([])
  const [metrics, setMetrics] = useState<Metrics>(null)
  const [file, setFile] = useState<File | null>(null)
  const [scanMetadata, setScanMetadata] = useState<ScanMetadata | null>(null)
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [originalPreviewUrl, setOriginalPreviewUrl] = useState<string | null>(null)
  const [showOriginal, setShowOriginal] = useState(false)
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [contrast, setContrast] = useState(112)
  const [level, setLevel] = useState(100)
  const [zoom, setZoom] = useState(100)
  const [inverted, setInverted] = useState(false)
  const [rotation, setRotation] = useState(0)
  const [datasetPath, setDatasetPath] = useState('')
  const [epochs, setEpochs] = useState(10)
  const [maxTrainSamplesPerClass, setMaxTrainSamplesPerClass] = useState(500)
  const [architecture, setArchitecture] = useState<'compact_cnn' | 'resnet18_transfer'>('resnet18_transfer')
  const [studyId, setStudyId] = useState('')
  const [sequence, setSequence] = useState('Unknown')
  const [laterality, setLaterality] = useState('Not specified')
  const [quality, setQuality] = useState('Unreviewed')
  const [artifactLevel, setArtifactLevel] = useState('Not assessed')
  const [anatomicalCoverage, setAnatomicalCoverage] = useState('Not assessed')
  const [annotation, setAnnotation] = useState('')
  const [autoCrop, setAutoCrop] = useState(true)
  const [noiseReduction, setNoiseReduction] = useState(true)
  const [skullStripping, setSkullStripping] = useState(false)
  const [sliceIndex, setSliceIndex] = useState(0)
  const fileInput = useRef<HTMLInputElement>(null)
  const viewerRef = useRef<HTMLDivElement>(null)

  async function refresh() {
    const [statusResult, casesResult, metricsResult, datasetResult, researchResult] = await Promise.allSettled([
      fetch('/api/status').then((response) => response.json() as Promise<ApiStatus>),
      fetch('/api/cases').then((response) => response.json() as Promise<CaseRecord[]>),
      fetch('/api/metrics').then((response) => response.json() as Promise<{ metrics: Metrics }>),
      fetch('/api/dataset').then((response) => response.json() as Promise<DatasetStatus>),
      fetch('/api/datasets/research').then((response) => response.json() as Promise<{ sources: ResearchDataset[] }>),
    ])
    if (statusResult.status === 'fulfilled') setStatus(statusResult.value)
    if (casesResult.status === 'fulfilled') setCases(casesResult.value)
    if (metricsResult.status === 'fulfilled') setMetrics(metricsResult.value.metrics)
    if (datasetResult.status === 'fulfilled') {
      setDataset(datasetResult.value)
      if (datasetResult.value.ready) setDatasetPath((currentPath) => currentPath || datasetResult.value.path)
    }
    if (researchResult.status === 'fulfilled') setResearchDatasets(researchResult.value.sources)
  }

  const refreshFromEffect = useEffectEvent(() => {
    void refresh()
  })

  useEffect(() => {
    const initialRefresh = window.setTimeout(refreshFromEffect, 0)
    const timer = window.setInterval(refreshFromEffect, 5000)
    return () => {
      window.clearTimeout(initialRefresh)
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    if (!originalPreviewUrl) return
    return () => URL.revokeObjectURL(originalPreviewUrl)
  }, [originalPreviewUrl])

  async function selectFile(nextFile?: File) {
    if (!nextFile) return
    const isDicom = /\.(dcm|dicom|ima)$/i.test(nextFile.name)
    const isNifti = /\.(nii|nii\.gz)$/i.test(nextFile.name)
    if (!['image/jpeg', 'image/png'].includes(nextFile.type) && !isDicom && !isNifti) {
      setMessage('Choose an MR DICOM, NIfTI, JPG, or PNG image.')
      return
    }
    if (nextFile.size > 20 * 1024 * 1024) {
      setMessage('This image is larger than the 20 MB limit.')
      return
    }
    setFile(nextFile)
    setScanMetadata(null)
    setAnalysis(null)
    setMessage('')
    setBusy(true)
    setShowOriginal(false)
    const originalUrl = isDicom ? null : URL.createObjectURL(nextFile)
    setOriginalPreviewUrl(originalUrl)
    setPreviewUrl(originalUrl)
    try {
      const body = new FormData()
      body.append('file', nextFile)
      body.append('auto_crop', String(autoCrop))
      body.append('noise_reduction', String(noiseReduction))
      body.append('skull_stripping', String(skullStripping))
      const response = await fetch('/api/preprocess', { method: 'POST', body, headers: { 'X-User-Role': role } })
      if (response.ok) {
        const result = await response.json() as { preview_data_url: string; scan_metadata: ScanMetadata; quality_metrics?: AnalysisResult['quality_metrics'] }
        setPreviewUrl(result.preview_data_url)
        setScanMetadata(result.scan_metadata)
      }
      const caseResponse = await fetch('/api/cases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: nextFile.name }),
      })
      if (caseResponse.ok) {
        const record = await caseResponse.json() as CaseRecord
        setCases((previous) => [record, ...previous.filter((item) => item.id !== record.id)].slice(0, 100))
      }
      if (!response.ok) setMessage('Image preview is local. Start the API to enable preprocessing.')
    } catch {
      setMessage('Image preview is local. Start the API to enable preprocessing and case history.')
    } finally {
      setBusy(false)
    }
  }

  async function analyze() {
    if (!file || !status.model_ready) return
    setBusy(true)
    try {
      const body = new FormData()
      body.append('file', file)
      const response = await fetch('/api/analyze', { method: 'POST', body, headers: { 'X-User-Role': role } })
      const result = await response.json() as AnalysisResult & { detail?: string }
      if (response.ok) {
        setAnalysis(result)
        setMessage('Analysis completed. Review all findings with a qualified radiologist.')
      } else {
        setAnalysis(null)
        setMessage(result.detail ?? 'Analysis could not be completed.')
      }
      await refresh()
    } catch {
      setMessage('Could not reach the analysis API.')
    } finally {
      setBusy(false)
    }
  }

  async function startTraining() {
    setBusy(true)
    setMessage('')
    try {
      const response = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_path: datasetPath, epochs, architecture, max_train_samples_per_class: maxTrainSamplesPerClass }),
      })
      const result = await response.json() as { detail?: string; message?: string }
      setMessage(response.ok ? result.message ?? 'Training started.' : result.detail ?? 'Training could not be started.')
      await refresh()
    } catch {
      setMessage('Could not reach the training API.')
    } finally {
      setBusy(false)
    }
  }

  async function preparePublicDataset() {
    setBusy(true)
    setMessage('')
    try {
      const response = await fetch('/api/dataset/prepare', { method: 'POST' })
      const result = await response.json() as { detail?: string; message?: string }
      setMessage(response.ok ? result.message ?? 'Dataset preparation started.' : result.detail ?? 'Dataset preparation could not be started.')
      await refresh()
    } catch {
      setMessage('Could not reach the dataset preparation API.')
    } finally {
      setBusy(false)
    }
  }

  async function copyMLInstallCommand() {
    const command = 'pip install -r requirements-ml.txt'
    try {
      await navigator.clipboard.writeText(command)
      setMessage('Install command copied. Run it in the project environment, then restart the API.')
    } catch {
      setMessage(`Run this in the project environment, then restart the API: ${command}`)
    }
  }

  function exportReport() {
    if (!metrics) return
    const url = URL.createObjectURL(new Blob([JSON.stringify({ disclaimer: 'For research and decision support only. Not a diagnosis.', metrics }, null, 2)], { type: 'application/json' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'model-evaluation-report.json'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  function exportResearchReport() {
    if (!file) return
    const report = {
      report_type: 'neuroimaging_research_review',
      disclaimer: 'Research annotation and model output only. Not a diagnosis and not for clinical decision-making.',
      exported_at: new Date().toISOString(),
      study: {
        local_id: studyId || file.name,
        filename: file.name,
        sequence,
        laterality,
        scan_metadata: scanMetadata,
      },
      quality_review: { quality, artifact_level: artifactLevel, anatomical_coverage: anatomicalCoverage },
      structured_annotation: annotation,
      research_prediction: analysis,
      robustness_check: analysis?.robustness_check,
      viewer: { contrast, brightness: level, zoom, inverted, rotation },
    }

    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${(studyId || file.name).replace(/\.[^/.]+$/, '')}-research-review.json`
    anchor.click()
    URL.revokeObjectURL(url)
    setMessage('Research review exported locally.')
  }

  function printDoctorReport() {
    if (!file) return
    window.print()
  }

  async function toggleFullscreen() {
    const viewer = viewerRef.current
    if (!viewer) return
    if (document.fullscreenElement === viewer) {
      await document.exitFullscreen()
    } else {
      await viewer.requestFullscreen()
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#review" onClick={() => setScreen('review')}>
          <span className="brand-mark"><Brain size={21} strokeWidth={1.8} /></span>
          <span><strong>NEURO</strong><small>IMAGING LAB</small></span>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="nav-list" aria-label="Main navigation">
          <button className={screen === 'review' ? 'nav-item active' : 'nav-item'} onClick={() => setScreen('review')}><ScanLine size={17} />Review studies<span className="nav-count">{cases.length}</span></button>
          <button className={screen === 'training' ? 'nav-item active' : 'nav-item'} onClick={() => setScreen('training')}><Layers3 size={17} />Model training</button>
          <button className={screen === 'evaluation' ? 'nav-item active' : 'nav-item'} onClick={() => setScreen('evaluation')}><Gauge size={17} />Evaluation</button>
          <button className={screen === 'dashboard' ? 'nav-item active' : 'nav-item'} onClick={() => setScreen('dashboard')}><ClipboardCheck size={17} />Doctor dashboard</button>
        </nav>
        <div className="sidebar-bottom">
          <div className="privacy-note"><ShieldCheck size={17} /><span><strong>Local-first workspace</strong><small>Images stay on this machine</small></span></div>
          <div className="profile-row"><div className="profile-avatar">NL</div><span><strong>Neuro Lab</strong><small>Research workspace</small></span><Menu size={17} /></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><ChevronRight size={14} /><strong>{screen === 'review' ? 'Study review' : screen === 'training' ? 'Model training' : screen === 'dashboard' ? 'Doctor dashboard' : 'Evaluation'}</strong></div>
          <div className="topbar-right"><label className="role-picker">ROLE<select value={role} onChange={(event) => setRole(event.target.value as typeof role)}><option value="radiologist">Radiologist</option><option value="admin">Admin</option><option value="patient">Patient</option></select></label><span className={status.api_ready ? 'system-status online' : 'system-status'}><i />{status.api_ready ? 'API connected' : 'API offline'}</span><button className="help-button" title="About this research tool" aria-label="About this research tool"><CircleHelp size={18} /></button></div>
        </header>

        <div className="page-content">
          <div className="page-heading">
            <div><div className="eyebrow">NEUROIMAGING · DECISION SUPPORT</div><h1>{screen === 'review' ? 'Study review' : screen === 'training' ? 'Model training' : screen === 'dashboard' ? 'Doctor dashboard' : 'Model evaluation'}</h1><p>{screen === 'review' ? 'Review MRI studies and inspect image preprocessing.' : screen === 'training' ? 'Train a research model using a local, labeled dataset.' : screen === 'dashboard' ? 'Review history, annotations, and printable research reports.' : 'Review validation performance from the latest training run.'}</p></div>
            <div className={status.model_ready ? 'model-pill ready' : 'model-pill'}><span className="model-dot" /><span><small>ACTIVE MODEL</small><strong>{status.model_ready ? status.model_name : 'No trained model'}</strong></span><ChevronRight size={15} /></div>
          </div>

          {screen === 'training' && <div className="architecture-selector"><label htmlFor="architecture">MODEL ARCHITECTURE</label><select id="architecture" value={architecture} onChange={(event) => setArchitecture(event.target.value as 'compact_cnn' | 'resnet18_transfer')}><option value="compact_cnn">Compact CNN · train from scratch</option><option value="resnet18_transfer">ResNet-18 · transfer learning</option></select><span>{architecture === 'resnet18_transfer' ? 'Downloads pretrained ImageNet weights on the first run; still requires MRI-specific training and validation.' : 'Lightweight grayscale CNN trained from scratch on your local dataset.'}</span></div>}

          {message && <div className="notice" role="status"><AlertCircle size={17} /><span>{message}</span><button onClick={() => setMessage('')} aria-label="Dismiss message"><X size={16} /></button></div>}

          {!status.model_ready && <section className="model-guidance" aria-label="Model setup status"><span className="guidance-icon"><Brain size={19} /></span><div className="guidance-copy"><strong>No trained MRI model is loaded</strong><p>Image preview and preprocessing work now. A tumor classification cannot be produced until a model is trained on a labeled MRI dataset.</p><small>{status.ml_dependencies_ready ? 'Training runtime is installed; choose your dataset under Model training.' : 'Training runtime is not installed. Install the optional ML packages, then restart the API.'}</small></div><div className="guidance-actions">{!status.ml_dependencies_ready && <button className="secondary-button" onClick={() => void copyMLInstallCommand()}>Copy setup command</button>}{screen !== 'training' && <button className="primary-button" onClick={() => setScreen('training')}>Open model training<ChevronRight size={14} /></button>}</div></section>}

          {screen === 'review' && <>
            <section className="stat-row" aria-label="Workspace summary">
              <article className="stat-tile"><span className="stat-icon teal"><FolderOpen size={17} /></span><div><small>STUDIES IN QUEUE</small><strong>{cases.length}</strong></div><span className="stat-caption">Local records</span></article>
              <article className="stat-tile"><span className="stat-icon coral"><Microscope size={17} /></span><div><small>MODEL STATUS</small><strong className="stat-text">{status.model_ready ? 'Ready' : 'Not configured'}</strong></div><span className="stat-caption">{status.model_ready ? 'Inference enabled' : 'No predictions'}</span></article>
              <article className="stat-tile"><span className="stat-icon gold"><Activity size={17} /></span><div><small>LAST EVALUATION</small><strong className="stat-text">{metrics ? new Date(metrics.evaluated_at).toLocaleDateString() : 'Not available'}</strong></div><span className="stat-caption">Held-out validation</span></article>
            </section>

            <section className="panel preprocessing-panel" aria-label="AI preprocessing controls">
              <div className="panel-heading"><div><h2>AI preprocessing</h2><p>Research controls sent to the local preprocessing pipeline.</p></div><Microscope size={18} className="subtle-icon" /></div>
              <div className="preprocessing-controls">
                <label><input type="checkbox" checked={autoCrop} onChange={(event) => setAutoCrop(event.target.checked)} /> Auto crop</label>
                <label><input type="checkbox" checked={noiseReduction} onChange={(event) => setNoiseReduction(event.target.checked)} /> Noise reduction</label>
                <label><input type="checkbox" checked={skullStripping} onChange={(event) => setSkullStripping(event.target.checked)} /> Research skull mask</label>
                <label>Slice selector<input type="range" min="0" max={Math.max(0, (scanMetadata?.slice_count ?? 1) - 1)} value={sliceIndex} onChange={(event) => setSliceIndex(Number(event.target.value))} disabled={(scanMetadata?.slice_count ?? 1) < 2} /><span>{scanMetadata?.slice_count ? `${sliceIndex + 1} / ${scanMetadata.slice_count}` : 'Single slice'}</span></label>
              </div>
              <small className="subtle-copy">NIfTI volumes are previewed one slice at a time. Skull masking and quality metrics are engineering aids, not clinical segmentation.</small>
            </section>

            <section className="panel research-tools-panel" aria-label="Structured neuroimaging review">
              <div className="panel-heading"><div><h2>Structured neuroimaging review</h2><p>Standardized research annotations saved only in the exported report</p></div><ClipboardCheck size={18} className="subtle-icon" /></div>
              <div className="research-form-grid">
                <label>LOCAL STUDY ID<input value={studyId} onChange={(event) => setStudyId(event.target.value)} placeholder="e.g. NRL-2026-001" /></label>
                <label>SEQUENCE<select value={sequence} onChange={(event) => setSequence(event.target.value)}><option>Unknown</option><option>T1</option><option>T1 post-contrast</option><option>T2</option><option>FLAIR</option><option>DWI</option><option>ADC</option></select></label>
                <label>LATERALITY<select value={laterality} onChange={(event) => setLaterality(event.target.value)}><option>Not specified</option><option>Left</option><option>Right</option><option>Midline</option><option>Bilateral</option></select></label>
                <label>IMAGE QUALITY<select value={quality} onChange={(event) => setQuality(event.target.value)}><option>Unreviewed</option><option>Diagnostic quality</option><option>Limited</option><option>Non-diagnostic</option></select></label>
                <label>ARTIFACT LEVEL<select value={artifactLevel} onChange={(event) => setArtifactLevel(event.target.value)}><option>Not assessed</option><option>None apparent</option><option>Mild</option><option>Significant</option></select></label>
                <label>ANATOMICAL COVERAGE<select value={anatomicalCoverage} onChange={(event) => setAnatomicalCoverage(event.target.value)}><option>Not assessed</option><option>Complete slice</option><option>Partial brain</option><option>Unclear</option></select></label>
              </div>
              <label className="annotation-field">RESEARCH OBSERVATION<textarea value={annotation} onChange={(event) => setAnnotation(event.target.value)} placeholder="Record neutral, image-grounded observations for research review. Do not enter patient identifiers." rows={2} /></label>
              <div className="research-tools-footer"><span><ShieldCheck size={14} /> De-identification is applied in memory; do not enter patient identifiers.</span><div className="footer-actions"><button className="secondary-button" disabled={!file} onClick={exportResearchReport}><FileText size={14} />Export JSON</button><button className="secondary-button" disabled={!file} onClick={printDoctorReport}><FileText size={14} />Print / PDF report</button></div></div>
            </section>

            <section className="review-grid">
              <div ref={viewerRef} className="panel study-panel">
                <div className="panel-heading"><div><h2>Current study</h2><p>{file ? 'Image preview · preprocessing applied when API is connected' : 'Add a scan to begin image review'}</p></div><button className="icon-button" title="Upload MRI image" aria-label="Upload MRI image" onClick={() => fileInput.current?.click()}><ImagePlus size={18} /></button></div>
                <input ref={fileInput} className="sr-only" type="file" accept="image/jpeg,image/png,application/dicom,application/octet-stream,.dcm,.dicom,.ima,.nii,.nii.gz" onChange={(event) => void selectFile(event.target.files?.[0])} />
                <div className={`scan-viewer ${dragging ? 'dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); void selectFile(event.dataTransfer.files[0]) }}>
                  {previewUrl ? <>
                    <div className="image-frame"><img src={showOriginal && originalPreviewUrl ? originalPreviewUrl : previewUrl} alt={showOriginal ? 'Original MRI image' : 'Enhanced MRI image preview'} style={{ filter: `contrast(${contrast}%) brightness(${level}%) ${inverted ? 'invert(1)' : ''}`, transform: `rotate(${rotation}deg) scale(${zoom / 100})` }} /></div>
                    <div className="image-overlay-label"><span><i />{showOriginal ? 'ORIGINAL' : 'PREVIEW'}</span><span>{scanMetadata?.modality ?? 'IMAGE'} · 2D</span></div>
                    {busy && <div className="image-loading"><LoaderCircle size={21} className="spin" />Processing image</div>}
                  </> : <button className="upload-prompt" onClick={() => fileInput.current?.click()}><span className="upload-icon"><Upload size={20} /></span><strong>Drop an MRI image here</strong><span>or browse DICOM, JPG, and PNG</span><small>Single-frame MR DICOM · 20 MB maximum</small></button>}
                </div>
                <div className="viewer-controls">
                  <div className="viewer-sliders">
                    <div className="control-group"><Contrast size={15} /><label htmlFor="contrast">Contrast</label><input id="contrast" type="range" min="50" max="200" value={contrast} onChange={(event) => setContrast(Number(event.target.value))} /><span>{contrast}%</span></div>
                    <div className="control-group"><Sun size={15} /><label htmlFor="level">Brightness</label><input id="level" type="range" min="50" max="150" value={level} onChange={(event) => setLevel(Number(event.target.value))} /><span>{level}%</span></div>
                    <div className="control-group"><span className="zoom-glyph">+</span><label htmlFor="zoom">Zoom</label><input id="zoom" type="range" min="75" max="200" value={zoom} onChange={(event) => setZoom(Number(event.target.value))} /><span>{zoom}%</span></div>
                  </div>
                  <div className="viewer-tools">
                    <button className={showOriginal ? 'tool-button selected' : 'tool-button'} title="Show original image" aria-label="Show original image" aria-pressed={showOriginal} disabled={!originalPreviewUrl || busy} onClick={() => setShowOriginal(true)}><FileImage size={15} /><span>Original</span></button>
                    <button className={!showOriginal ? 'tool-button selected' : 'tool-button'} title="Show preprocessed image" aria-label="Show enhanced image" aria-pressed={!showOriginal} disabled={!previewUrl || busy} onClick={() => setShowOriginal(false)}><Layers3 size={15} /><span>Enhanced</span></button>
                    <button className={inverted ? 'tool-button selected' : 'tool-button'} title="Invert grayscale" aria-label="Invert grayscale" aria-pressed={inverted} disabled={!previewUrl} onClick={() => setInverted((value) => !value)}><Contrast size={15} /><span>Invert</span></button>
                    <button className="tool-button" title="Rotate 90 degrees clockwise" aria-label="Rotate image" disabled={!previewUrl} onClick={() => setRotation((value) => (value + 90) % 360)}><RotateCw size={15} /><span>Rotate</span></button>
                    <button className="tool-button" title="Reset viewer controls" aria-label="Reset viewer controls" disabled={!previewUrl} onClick={() => { setContrast(112); setLevel(100); setZoom(100); setInverted(false); setRotation(0) }}><RotateCcw size={15} /><span>Reset</span></button>
                    <button className="tool-button" title="Open fullscreen viewer" aria-label="Open fullscreen viewer" disabled={!previewUrl} onClick={() => void toggleFullscreen()}><Maximize2 size={15} /><span>Full screen</span></button>
                  </div>
                </div>
                <div className="study-footer"><div className="file-info"><FileImage size={16} /><span><strong>{file?.name ?? 'No image selected'}</strong><small>{file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB · ${scanMetadata?.format ?? file.type.split('/')[1]?.toUpperCase() ?? 'MRI'}` : 'MR DICOM, JPG, or PNG'}</small></span></div><button className="text-button" disabled={!file} onClick={() => { setFile(null); setScanMetadata(null); setAnalysis(null); setPreviewUrl(null); setOriginalPreviewUrl(null); setShowOriginal(false); setMessage('') }}><X size={15} />Clear</button></div>
              </div>

              <div className="right-column">
                <section className="panel result-panel"><div className="panel-heading"><div><h2>Research prediction</h2><p>Model output for research review</p></div><span className="locked-icon"><ShieldCheck size={17} /></span></div>{analysis ? <div className="analysis-result"><div className="prediction-heading"><span className="result-mark"><Brain size={22} /></span><div><small>PREDICTED CLASS</small><strong>{analysis.predicted_class.replaceAll('_', ' ')}</strong></div><span className="confidence-value">{(analysis.confidence * 100).toFixed(1)}%</span></div>{analysis.uncertainty.flagged && <div className="uncertainty-warning"><AlertCircle size={14} /><span>Research uncertainty flag: {analysis.uncertainty.reason}.</span></div>}{analysis.tumor_present !== null && <div className="finding-row"><span>Model output</span><strong className={analysis.tumor_present ? 'finding-positive' : 'finding-negative'}>{analysis.tumor_present ? 'Tumor class predicted' : 'No-tumor class predicted'}</strong></div>}<div className="probability-list"><strong>Class probabilities</strong>{Object.entries(analysis.probabilities).map(([name, probability]) => <div className="probability-row" key={name}><span>{name.replaceAll('_', ' ')}</span><div><i style={{ width: `${Math.min(100, probability * 100)}%` }} /></div><small>{(probability * 100).toFixed(1)}%</small></div>)}</div><div className="research-quality"><strong>Automated image-quality screen</strong><span>Score {analysis.quality_metrics.score}/100 · {analysis.quality_metrics.flags.length ? analysis.quality_metrics.flags.join(', ') : 'no heuristic flags'}</span><small>{analysis.quality_metrics.interpretation}</small></div>                <div className="robustness-check"><strong>Prediction robustness stress test</strong><span>{analysis.robustness_check.stable ? 'Stable across mild intensity changes' : 'Sensitive to mild intensity changes'} · {Math.round(analysis.robustness_check.agreement * 100)}% agreement</span><small>{analysis.robustness_check.interpretation}</small></div><div className="segmentation-status"><strong>Segmentation and severity</strong><span>{analysis.segmentation?.available ? `Mask volume ${analysis.segmentation.volume_cm3 ?? '—'} cm³` : 'Not configured'}</span><small>{analysis.severity_grade?.reason ?? analysis.segmentation?.reason ?? 'A validated segmentation model is required.'}</small></div><div className="heatmap-heading"><span>Grad-CAM · exploratory</span><span>Not a segmentation</span></div><img className="heatmap-preview" src={analysis.heatmap_data_url} alt="Exploratory Grad-CAM heatmap overlay" /><p className="explanation-note">{analysis.explanation_note} Model: {analysis.model_provenance.model_name ?? 'unknown'}.</p></div> : <div className="result-placeholder"><span className="result-mark"><Brain size={23} /></span><strong>{status.model_ready ? 'Ready for research prediction' : 'Training required'}</strong><p>{status.model_ready ? 'A trained model is available. Results require professional review.' : 'No prediction is being made. Follow the model setup status above to enable real inference.'}</p></div>}<div className="analysis-action"><button className="primary-button" disabled={!file || !status.model_ready || !status.ml_dependencies_ready || busy} onClick={() => void analyze()}>{busy ? <LoaderCircle size={16} className="spin" /> : <ScanLine size={16} />}Generate research prediction</button></div><div className="disclaimer"><AlertCircle size={15} /><span>{analysis?.disclaimer ?? 'Research decision support only. Not a diagnosis.'}</span></div></section>
                <section className="panel details-panel">
                  <div className="panel-heading"><div><h2>Image details</h2><p>Technical metadata only</p></div><HardDrive size={17} className="subtle-icon" /></div>
                  <dl className="detail-list">
                    <div><dt>File name</dt><dd>{file?.name ?? '—'}</dd></div>
                    <div><dt>Format</dt><dd>{scanMetadata?.format ?? '—'}</dd></div>
                    <div><dt>Modality</dt><dd>{scanMetadata?.modality ?? 'Not provided'}</dd></div>
                    <div><dt>Dimensions</dt><dd>{scanMetadata ? `${scanMetadata.columns} × ${scanMetadata.rows} px` : '—'}</dd></div>
                    <div><dt>Pixel spacing</dt><dd>{scanMetadata?.pixel_spacing_mm ? `${scanMetadata.pixel_spacing_mm.join(' × ')} mm` : 'Not provided'}</dd></div>
                    <div><dt>Model</dt><dd>{status.model_ready ? status.model_name : 'Not available'}</dd></div>
                    <div><dt>Preprocessing</dt><dd><span className="inline-status"><i />{status.api_ready && file ? 'Enhanced preview' : 'Waiting for image'}</span></dd></div>
                  </dl>
                </section>
              </div>
            </section>

            {analysis && <ConditionGuidance predictedClass={analysis.predicted_class} />}

            <section className="panel queue-panel"><div className="panel-heading"><div><h2>Recent studies</h2><p>Local review history · image data is not retained</p></div><span className="queue-total">{cases.length} RECORDS</span></div>{cases.length ? <div className="table-wrap"><table><thead><tr><th>FILE</th><th>ADDED</th><th>STATUS</th><th aria-label="Open study" /></tr></thead><tbody>{cases.slice(0, 5).map((item) => <tr key={item.id}><td><span className="table-file"><FileImage size={15} />{item.filename}</span></td><td>{new Date(item.created_at).toLocaleString()}</td><td><span className="queue-status"><i />Preview only</span></td><td><button className="row-action" title={`Load ${item.filename}`} aria-label={`Load ${item.filename}`} onClick={() => { setScreen('review'); setMessage('Select the original image again to create a new preview.') }}><ChevronRight size={16} /></button></td></tr>)}</tbody></table></div> : <div className="empty-queue"><span><FolderOpen size={18} /></span><div><strong>No studies in the queue</strong><small>Uploaded files are not retained after preview.</small></div></div>}</section>
          </>}

          {screen === 'dashboard' && <section className="dashboard-layout">
            <section className="panel dashboard-card"><div className="panel-heading"><div><h2>Patient history timeline</h2><p>Local filenames and timestamps only; no patient identity is stored.</p></div><FolderOpen size={18} className="subtle-icon" /></div>{cases.length ? <div className="timeline">{cases.map((item) => <div className="timeline-item" key={item.id}><span className="timeline-dot" /><div><strong>{item.filename}</strong><small>{new Date(item.created_at).toLocaleString()} · {item.status}</small></div></div>)}</div> : <p className="subtle-copy">Upload a study to begin the local timeline.</p>}</section>
            <section className="panel dashboard-card"><div className="panel-heading"><div><h2>Radiologist workspace</h2><p>Structured annotation is available in Study review.</p></div><ClipboardCheck size={18} className="subtle-icon" /></div><p className="dashboard-note">Use the role selector to identify the current local workspace role. This prototype does not provide production authentication or clinical access control.</p><button className="primary-button" onClick={() => setScreen('review')}><ScanLine size={15} />Open study review</button></section>
            <section className="panel dashboard-card"><div className="panel-heading"><div><h2>Report generator</h2><p>Print the current research review to a PDF using your browser.</p></div><FileText size={18} className="subtle-icon" /></div><p className="dashboard-note">Reports are labeled research-only and should not be used as diagnostic documents.</p><button className="secondary-button" disabled={!file} onClick={printDoctorReport}><FileText size={15} />Print current report</button></section>
          </section>}

          {screen === 'training' && <>
            <section className="panel sample-cap-panel"><div className="panel-heading"><div><h2>Training sample limit</h2><p>Set a per-class cap to keep local CPU runs practical.</p></div></div><div className="sample-cap-control"><label className="field-label" htmlFor="sample-cap">MAX TRAIN IMAGES PER CLASS</label><input id="sample-cap" type="number" min="1" max="10000" step="100" value={maxTrainSamplesPerClass} onChange={(event) => setMaxTrainSamplesPerClass(Math.min(10000, Math.max(1, Number(event.target.value) || 1)))} /><span>Official test images remain uncapped.</span></div></section>
            <section className="panel dataset-source-panel"><div className="dataset-source-icon"><HardDrive size={19} /></div><div className="dataset-source-copy"><div className="dataset-source-title"><h2>Public research dataset</h2><span>CC0-1.0</span></div><p>7,023 four-class brain MRI images from Hugging Face. Downloads to this machine; official train and test splits stay separate.</p><small>Classes: glioma, meningioma, no tumor, pituitary. Source card does not document patient-level separation.</small>{dataset.error && <strong className="dataset-error">{dataset.error}</strong>}{dataset.message && <small className="dataset-message">{dataset.message}</small>}</div><div className="dataset-source-actions">{dataset.ready ? <><span className="dataset-ready"><Check size={14} />{dataset.metadata?.image_count.toLocaleString()} images ready</span><button className="secondary-button" onClick={() => setDatasetPath(dataset.path)}>Use this dataset</button></> : <button className="primary-button" disabled={!status.api_ready || dataset.preparing || busy} onClick={() => void preparePublicDataset()}>{dataset.preparing ? <LoaderCircle size={15} className="spin" /> : <Download size={15} />}{dataset.preparing ? 'Preparing dataset' : 'Download and prepare'}</button>}</div></section>
            <section className="panel research-datasets-panel"><div className="panel-heading"><div><h2>Referenced MRI datasets</h2><p>Research sources reviewed for dataset selection and reporting.</p></div><Microscope size={18} className="subtle-icon" /></div><div className="research-dataset-grid">{researchDatasets.map((source) => <article className="research-dataset-card" key={source.id}><div className="research-dataset-card-heading"><div><h3>{source.name}</h3><span>{source.publisher}</span></div><a href={source.url} target="_blank" rel="noreferrer">Open source</a></div><p>{source.split_summary}{source.image_count ? ` · ${source.image_count.toLocaleString()} images` : ''}</p><div className="research-dataset-tags">{source.classes.length ? source.classes.map((item) => <span key={item}>{item.replace('_', ' ')}</span>) : <span>Verification needed</span>}<span>{source.license}</span></div><ul>{source.notes.map((note) => <li key={note}>{note}</li>)}</ul></article>)}</div></section>
            <section className="training-layout"><div className="panel training-panel"><div className="panel-heading"><div><h2>Configure training</h2><p>Dataset folders should contain one subfolder per class.</p></div><Layers3 size={18} className="subtle-icon" /></div><div className="training-content"><label className="field-label" htmlFor="dataset-path">LOCAL DATASET PATH</label><div className="path-input"><FolderOpen size={17} /><input id="dataset-path" value={datasetPath} onChange={(event) => setDatasetPath(event.target.value)} placeholder="D:\datasets\brain-mri" /><span className="path-tag">LOCAL</span></div><div className="folder-example"><code>brain-mri/</code><span>→</span><code>glioma/</code><code>meningioma/</code><code>no_tumor/</code><span>...</span></div><div className="field-label epoch-label">TRAINING EPOCHS</div><div className="epoch-setting"><input type="range" min="5" max="50" step="5" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} /><strong>{epochs}</strong><span>epochs</span></div><div className="training-note"><AlertCircle size={16} /><span>Training runs locally and can take several minutes. Use a labeled dataset with patient-level separation to prevent data leakage.</span></div><button className="primary-button training-submit" disabled={!datasetPath.trim() || busy || status.training || dataset.preparing} onClick={() => void startTraining()}>{busy || status.training ? <LoaderCircle size={16} className="spin" /> : <Activity size={16} />}{status.training ? 'Training in progress' : 'Start training'}</button>{status.training_message && <p className="training-progress">{status.training_message}</p>}</div></div><aside className="training-aside"><div className="aside-icon"><ShieldCheck size={20} /></div><h3>Model safety</h3><p>This prototype does not ship with pretrained weights. Training results are experimental and must be independently validated before any clinical use.</p><div className="aside-divider" /><div className="aside-row"><span>API</span><strong>{status.api_ready ? 'Connected' : 'Offline'}</strong></div><div className="aside-row"><span>ML runtime</span><strong>{status.ml_dependencies_ready ? 'Ready' : 'Missing'}</strong></div><div className="aside-row"><span>Model</span><strong>{status.model_ready ? 'Loaded' : 'Not loaded'}</strong></div><div className="aside-row"><span>Classes</span><strong>{status.classes.length || '—'}</strong></div></aside></section>
          </>}

          {screen === 'evaluation' && <section className="panel evaluation-panel"><div className="panel-heading"><div><h2>Validation metrics</h2><p>Calculated from the latest held-out evaluation.</p></div><button className="icon-button" title="Download report" aria-label="Download report" onClick={exportReport} disabled={!metrics}><ArrowDownToLine size={17} /></button></div>{metrics ? <><div className="metric-grid">{[['Accuracy', metrics.accuracy], ['Precision', metrics.precision], ['Recall', metrics.recall], ['F1 score', metrics.f1_score]].map(([label, value]) => <div className="metric-card" key={label as string}><span>{label}</span><strong>{(Number(value) * 100).toFixed(1)}<small>%</small></strong></div>)}</div><div className="confusion-wrap"><h3>Confusion matrix</h3><table className="confusion-table"><thead><tr><th>Actual \ Predicted</th>{metrics.classes.map((name) => <th key={name}>{name}</th>)}</tr></thead><tbody>{metrics.confusion_matrix.map((row, index) => <tr key={metrics.classes[index]}><th>{metrics.classes[index]}</th>{row.map((count, column) => <td className={index === column ? 'diagonal-cell' : ''} key={`${index}-${column}`}>{count}</td>)}</tr>)}</tbody></table></div><p className="evaluation-time"><Check size={15} />Evaluated {new Date(metrics.evaluated_at).toLocaleString()}</p></> : <div className="empty-evaluation"><span><Gauge size={22} /></span><strong>No evaluation results yet</strong><p>Train a model with a labeled dataset to generate validation metrics and a confusion matrix.</p><button className="secondary-button" onClick={() => setScreen('training')}>Configure training<ChevronRight size={15} /></button></div>}<div className="disclaimer evaluation-disclaimer"><AlertCircle size={15} /><span>Metrics are dataset-dependent and do not establish clinical safety or diagnostic accuracy.</span></div></section>}

          <footer className="page-footer"><span><ShieldCheck size={14} />Local research prototype</span><span>Not for diagnosis or treatment decisions</span></footer>
        </div>
      </main>
    </div>
  )
}

export default App
