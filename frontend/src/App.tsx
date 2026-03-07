import { useState, useEffect, useRef } from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Divider } from '@mui/material'
import { faEye } from '@fortawesome/free-solid-svg-icons'
import './App.css'

interface VADPoint {
  t: number
  confidence: number
}

interface LoggerEvent {
  id: number | string
  ts: number
  source: string
  severity: string
  message: string
}

interface Toast {
  id: number
  source: string
  message: string
  ts: number
  severity: string
}

function App() {
  const [pipelineEnabled, setPipelineEnabled] = useState<boolean | null>(null)
  const [vadStatus, setVadStatus] = useState('waiting...')
  const [output, setOutput] = useState<string>('{ "status": "checking_pipeline_status" }')
  const [points, setPoints] = useState<VADPoint[]>([])
  const [loggerEvents, setLoggerEvents] = useState<LoggerEvent[]>([])
  const [toasts, setToasts] = useState<Toast[]>([])
  const [avgText, setAvgText] = useState('Rolling average: waiting...')
  
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const eventSourceRef = useRef<EventSource | null>(null)
  const toastCounterRef = useRef(0)
  const loggerEventCounterRef = useRef(0)

  const AVG_WINDOW = 10
  const MAX_POINTS = 300
  const MAX_LOGGER_EVENTS = 5

  const pushPoint = (t: number, confidence: number) => {
    if (!Number.isFinite(t) || !Number.isFinite(confidence)) return
    setPoints(prev => {
      const updated = [...prev, { t, confidence }]
      while (updated.length > MAX_POINTS) updated.shift()
      return updated
    })
  }

  const pushLoggerEvent = (alert: any) => {
    if (!alert || !alert.ts || !alert.message) return
    const newEvent: LoggerEvent = {
      id: `event-${loggerEventCounterRef.current++}`,
      ts: alert.ts,
      source: alert.source || 'unknown',
      severity: alert.severity || 'info',
      message: alert.message,
    }
    setLoggerEvents(prev => {
      const updated = [newEvent, ...prev]
      while (updated.length > MAX_LOGGER_EVENTS) updated.pop()
      return updated
    })
  }

  const showToast = (alert: any) => {
    const newToast: Toast = {
      id: toastCounterRef.current++,
      source: (alert.source || 'SRC').toUpperCase(),
      message: alert.message,
      ts: alert.ts,
      severity: alert.severity || 'info',
    }
    setToasts(prev => [...prev, newToast])
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== newToast.id))
    }, 6000)
  }

  const rollingAvgLastN = (n: number): number | null => {
    if (points.length === 0) return null
    const k = Math.max(1, Math.min(points.length, n))
    let sum = 0
    for (let i = points.length - k; i < points.length; i++) {
      sum += points[i].confidence
    }
    return sum / k
  }

  const buildRollingAvgSeries = (n: number): VADPoint[] => {
    const series: VADPoint[] = []
    const k = Math.max(1, n)
    for (let i = 0; i < points.length; i++) {
      const start = Math.max(0, i - k + 1)
      let sum = 0
      let count = 0
      for (let j = start; j <= i; j++) {
        sum += points[j].confidence
        count += 1
      }
      series.push({ t: points[i].t, confidence: sum / count })
    }
    return series
  }

  const drawChart = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    const padL = 52,
      padR = 12,
      padT = 10,
      padB = 26
    const w = canvas.width - padL - padR
    const h = canvas.height - padT - padB

    ctx.strokeStyle = '#cccccc'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(padL, padT)
    ctx.lineTo(padL, padT + h)
    ctx.lineTo(padL + w, padT + h)
    ctx.stroke()

    ctx.fillStyle = '#333333'
    ctx.font = '12px Arial'
    const yTicks = [0, 0.5, 1.0]
    yTicks.forEach(v => {
      const y = padT + (1 - v) * h
      ctx.strokeStyle = '#eeeeee'
      ctx.beginPath()
      ctx.moveTo(padL, y)
      ctx.lineTo(padL + w, y)
      ctx.stroke()

      ctx.fillStyle = '#333333'
      ctx.fillText(v.toFixed(1), 10, y + 4)
    })

    if (points.length < 2) {
      ctx.fillStyle = '#666'
      ctx.fillText('Waiting for VAD confidence...', padL + 10, padT + 18)
      return
    }

    const tMin = points[0].t
    const tMax = points[points.length - 1].t
    const tSpan = Math.max(1e-6, tMax - tMin)

    const leftLabel = new Date(tMin * 1000).toLocaleTimeString()
    const rightLabel = new Date(tMax * 1000).toLocaleTimeString()
    ctx.fillStyle = '#333333'
    ctx.fillText(leftLabel, padL, padT + h + 18)
    ctx.fillText(rightLabel, padL + w - ctx.measureText(rightLabel).width, padT + h + 18)

    ctx.strokeStyle = '#1f77b4'
    ctx.lineWidth = 2
    ctx.beginPath()
    for (let i = 0; i < points.length; i++) {
      const p = points[i]
      const x = padL + ((p.t - tMin) / tSpan) * w
      const y = padT + (1 - Math.max(0, Math.min(1, p.confidence))) * h
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    const avgSeries = buildRollingAvgSeries(AVG_WINDOW)
    ctx.strokeStyle = '#2ca02c'
    ctx.lineWidth = 2
    ctx.beginPath()
    for (let i = 0; i < avgSeries.length; i++) {
      const p = avgSeries[i]
      const x = padL + ((p.t - tMin) / tSpan) * w
      const y = padT + (1 - Math.max(0, Math.min(1, p.confidence))) * h
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    const avgNow = rollingAvgLastN(AVG_WINDOW)
    if (avgNow === null) {
      setAvgText('Rolling average: waiting...')
    } else {
      setAvgText(`Rolling average (last ${AVG_WINDOW}): ${avgNow.toFixed(3)}`)
    }
    const last = points[points.length - 1]
    const xLast = padL + ((last.t - tMin) / tSpan) * w
    const yLast = padT + (1 - Math.max(0, Math.min(1, last.confidence))) * h
    ctx.fillStyle = '#d62728'
    ctx.beginPath()
    ctx.arc(xLast, yLast, 4, 0, Math.PI * 2)
    ctx.fill()

    ctx.fillStyle = '#111'
    ctx.fillText(`latest: ${last.confidence.toFixed(3)}`, padL + 10, padT + 18)
  }

  useEffect(() => {
    drawChart()
  }, [points])

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch('/status')
        const data = await res.json()
        setPipelineEnabled(data.pipeline_enabled)

        if (!data.pipeline_enabled) {
          setVadStatus('VAD: pipeline disabled')
          setAvgText('Rolling average: pipeline disabled')
          setOutput(JSON.stringify({ status: 'pipeline_disabled' }, null, 2))
          return
        }

        // Load initial history
        try {
          const histRes = await fetch('/pipeline/history?limit=300')
          const hist = await histRes.json()
          if (hist.points && Array.isArray(hist.points)) {
            hist.points.forEach((p: VADPoint) => pushPoint(p.t, p.confidence))
          }
        } catch (e) {
          console.error('Failed to load history:', e)
        }

        // Connect to EventSource
        const es = new EventSource('/pipeline/stream')

        es.onopen = () => {
          setVadStatus('VAD: SSE connected...')
        }

        es.onmessage = (e: MessageEvent) => {
          try {
            const obj = JSON.parse(e.data)
            setOutput(JSON.stringify(obj, null, 2))

            if (obj && obj.vad) {
              const label = String(obj.vad.label || '').toUpperCase()
              const conf =
                typeof obj.vad.confidence === 'number'
                  ? obj.vad.confidence.toFixed(3)
                  : 'N/A'
                const cap = obj.vad.top_caption ? String(obj.vad.top_caption).slice(0, 60) : 'N/A'
              setVadStatus(`VAD: ${label} | confidence: ${conf} | caption: ${cap}`)
            }

            if (obj && obj.updated_at && obj.vad && typeof obj.vad.confidence === 'number') {
              pushPoint(obj.updated_at, obj.vad.confidence)
            }

            if (obj.alerts && obj.alerts.length) {
              obj.alerts.forEach((a: any) => {
                showToast(a)
                pushLoggerEvent(a)
              })
            }
          } catch (err) {
            setOutput(e.data)
          }
        }

        es.onerror = () => {
          setVadStatus('VAD: SSE disconnected — refresh page')
          setAvgText('Rolling average: SSE disconnected')
          setOutput(JSON.stringify({ error: 'SSE disconnected — refresh page' }, null, 2))
          es.close()
        }

        eventSourceRef.current = es
      } catch (err) {
        setPipelineEnabled(false)
        setOutput(JSON.stringify({ error: 'status check failed' }, null, 2))
        console.error('Status check failed:', err)
      }
    }

    checkStatus()

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
      }
    }
  }, [])

  return (
    <div className="app-container">
      <div className="top-bar">
        <FontAwesomeIcon icon={faEye} className='eye-icon' />
        <span className="app-title">Near Real-Time Knowledge-Guided Video Monitoring System</span>
      </div>

      {/* Dev only, delete when not needed  
      <div className="muted">
        Video: <code>/video/mjpeg</code> | Live VAD stream: <code>/pipeline/stream</code> | Metrics:{' '}
        <code>/frame-selector/metrics</code> | Overtime: <code>/pipeline/history</code>
      </div>
      */}

      {pipelineEnabled === false && (
        <div className="warn">
          <b>Pipeline is disabled.</b>
          This API is running, but ML dependencies are missing, so <code>/video/mjpeg</code> and{' '}
          <code>/pipeline/stream</code> will not be called.
        </div>
      )}

      <div className="dashboard-grid">
        <div className="card video-card">
          <h3>Live Video</h3>
          <div className="vad-badge">{vadStatus}</div>
          <div className='video-wrap' id="videoWrap">
            {pipelineEnabled && (
              <img id="videoImg" src="/video/mjpeg" alt="Live video stream" />
            )}
          </div>
        </div>
        <Divider orientation="vertical" style={{ gridArea: 'divider-v1', margin: '0 5px', backgroundColor: '#6235d4', width: '1px' }} />
        <div className="card output-card">
          <h3>Live VAD Output</h3>
          <pre id="out">{output}</pre>
        </div>
        <div className='divider-h-container' style={{ gridArea: 'eye-zone' }}>
          <Divider orientation="horizontal" className='divider-h' />
          <FontAwesomeIcon icon={faEye} className='eye-icon-center' />
          <Divider orientation="horizontal" className='divider-h' />
        </div>
        <div className="card chart-card">
          <h3>Overtime Performance VAD Only (Confidence vs Time)</h3>
          <div className="muted" id="avgLine" style={{ marginBottom: '8px' }}>
            {avgText}
          </div>
          <div className="muted" style={{ marginBottom: '8px' }}>
            X-axis = time, Y-axis = VAD confidence
          </div>
          <div className="muted" style={{ marginBottom: '8px' }}>
            blue = confidence vs time, green = rolling average
          </div>
          <div className="canvas-container">
            <canvas ref={canvasRef} id="perfChart" width={720} height={240} />
          </div>
        </div>
        <Divider orientation="vertical" style={{ gridArea: 'divider-v2', margin: '0 5px', backgroundColor: '#6235d4', width: '1px' }} />
        <div className="card events-card">
          <h3>Recent Events</h3>
          <div className="muted" style={{ marginBottom: '8px' }}>
            Last 5 Alerts
          </div>
          <div className="events-container">
            {loggerEvents.length === 0 ? (
              <div style={{ color: 'black', textAlign: 'center', padding: '20px' }}>
                No events yet...
              </div>
            ) : (
              loggerEvents.map(evt => {
                const time = new Date(evt.ts * 1000).toLocaleTimeString()
                const severityColor =
                  evt.severity === 'critical'
                    ? '#d62728'
                    : evt.severity === 'warning'
                      ? '#ff7f0e'
                      : '#1f77b4'
                return (
                  <div key={evt.id} className="event-item">
                    <div>
                      <span className="event-label" style={{ color: severityColor }}>
                        {evt.severity.toUpperCase()}
                      </span>{' '}
                      <span style={{ color: '#666', fontSize: '12px' }}>{evt.source}</span>
                    </div>
                    <div style={{ margin: '4px 0' }}>{evt.message}</div>
                    <div className="event-time">{time}</div>
                  </div>
                )
              })
            )}
          </div>
        </div>
      </div>

      {/* Toasts Container */}
      <div id="toasts" className="toasts">
        {toasts.map(toast => (
          <div
            key={toast.id}
            className={`toast ${toast.severity}`}
            onClick={() => setToasts(prev => prev.filter(t => t.id !== toast.id))}
          >
            <strong>{toast.source}</strong>: {toast.message}
            <div className="meta">
              {new Date(toast.ts * 1000).toLocaleTimeString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default App
