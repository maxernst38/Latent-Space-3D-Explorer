/**
 * The three.js side of the viewer: geometry, camera, picking, render loop.
 *
 * Deliberately free of React so the rendering can be reasoned about (and fixed)
 * without touching component state.
 *
 * Everything is driven by `update()` calls from the component. The scene owns
 * no application state of its own beyond what it needs to draw a frame.
 */

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { CENTROID_NEUTRAL, NEUTRAL, outlierColor, toRgbArray } from './palette.js'

const BACKGROUND = 0x3a3f49 // --color-canvas, which this must stay in step with
const BASE_POINT_SIZE = 3.1
const HIGHLIGHT_POINT_SIZE = 6.0
const DIMMED_ALPHA = 0.42
const PICK_THRESHOLD = 0.025

// Distance the camera starts at. Point sizes are quoted in CSS pixels *at this
// distance*, so `uScale` converts them into the device pixels gl_PointSize
// wants, and perspective divide handles zoom from there. Without a reference
// like this, `size` is implicitly in world units and a value of 2 fills the
// screen — the cloud spans only [-1, 1].
const REFERENCE_DISTANCE = 2.4

// Faint lines through the origin, for orientation while orbiting. Positions are
// normalised by the 98th-percentile radius at build time, so the origin is the
// cloud's centre and ±1 is roughly its edge in every direction — which is what
// makes a fixed extent a meaningful reference rather than an arbitrary one.
const AXIS_EXTENT = 1.15
const AXIS_COLOR = 0xaab4c4 // --color-canvas-muted
const AXIS_OPACITY = 0.22

// Round, soft-edged points with per-point size, colour and opacity. PointsMaterial
// can't vary size per point, which highlighting depends on, so this is the
// smallest shader that does the job.
const VERTEX_SHADER = `
  uniform float uScale;
  attribute float size;
  attribute float alpha;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vColor = color;
    vAlpha = alpha;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size * (uScale / max(-mv.z, 0.001));
    gl_Position = projectionMatrix * mv;
  }
`

const FRAGMENT_SHADER = `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 offset = gl_PointCoord - vec2(0.5);
    float sqDist = dot(offset, offset);
    if (sqDist > 0.25) discard;
    gl_FragColor = vec4(vColor, vAlpha * smoothstep(0.25, 0.14, sqDist));
  }
`

/**
 * The three world axes, crossing at 0,0,0.
 *
 * Not to be confused with the `showAxes` option, which draws one *class's* axis
 * of maximum variance. These are fixed, faint, and always on: a reference frame
 * rather than a reading of the data, which is why they render behind the cloud
 * and never pick up a highlight colour.
 */
function makeOriginAxes() {
  const e = AXIS_EXTENT
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(
      // prettier-ignore
      new Float32Array([
        -e, 0, 0,   e, 0, 0,
         0, -e, 0,  0, e, 0,
         0, 0, -e,  0, 0, e,
      ]),
      3,
    ),
  )

  const axes = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({
      color: AXIS_COLOR,
      transparent: true,
      opacity: AXIS_OPACITY,
      depthWrite: false,
    }),
  )
  // Drawn first, so the reference never reads as data sitting in front of data.
  axes.renderOrder = -1
  return axes
}

/** Shared by the point cloud and the centroid markers — same shader, same uniform. */
function makePointMaterial(pixelRatio) {
  return new THREE.ShaderMaterial({
    uniforms: { uScale: { value: REFERENCE_DISTANCE * pixelRatio } },
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    vertexColors: true,
  })
}

export function createScene(canvas) {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
    powerPreference: 'high-performance',
  })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(BACKGROUND)

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  camera.position.set(1.55, 1.02, 1.55)

  const controls = new OrbitControls(camera, canvas)
  controls.enableDamping = true
  controls.dampingFactor = 0.08
  controls.rotateSpeed = 0.7
  controls.minDistance = 0.4
  controls.maxDistance = 12
  controls.autoRotateSpeed = 0.55

  const raycaster = new THREE.Raycaster()
  raycaster.params.Points.threshold = PICK_THRESHOLD
  const pointer = new THREE.Vector2()

  const hullGroup = new THREE.Group()
  const axisGroup = new THREE.Group()
  // Built once rather than per bundle: every bundle is normalised into the same
  // space, so the reference frame never changes.
  const originAxes = makeOriginAxes()
  scene.add(hullGroup, axisGroup, originAxes)

  let bundle = null
  let points = null
  let centroids = null
  let highlights = new Map() // classId -> hex
  let options = {}
  let hovered = -1
  let dirty = true
  let running = true
  let frame = null
  let pointerActive = false

  let onHover = () => {}
  let onLabels = () => {}

  const invalidate = () => {
    dirty = true
  }
  controls.addEventListener('change', invalidate)

  /* ── Geometry ─────────────────────────────────────────────────────────── */

  function setBundle(next) {
    disposePoints()
    bundle = next
    if (!bundle) return invalidate()

    const count = bundle.manifest.count
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(bundle.positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(count * 3), 3))
    geometry.setAttribute('size', new THREE.BufferAttribute(new Float32Array(count), 1))
    geometry.setAttribute('alpha', new THREE.BufferAttribute(new Float32Array(count), 1))

    points = new THREE.Points(geometry, makePointMaterial(renderer.getPixelRatio()))
    // Points are added and removed as bundles change; frustum culling on a
    // cloud that fills the view costs more than it saves.
    points.frustumCulled = false
    scene.add(points)

    buildCentroids()
    applyStyling()
    resetCamera()
  }

  function buildCentroids() {
    const withCentroid = bundle.manifest.classes.filter((c) => c.centroid)
    const positions = new Float32Array(withCentroid.length * 3)
    withCentroid.forEach((entry, i) => positions.set(entry.centroid, i * 3))

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(withCentroid.length * 3), 3))
    geometry.setAttribute('size', new THREE.BufferAttribute(new Float32Array(withCentroid.length), 1))
    geometry.setAttribute('alpha', new THREE.BufferAttribute(new Float32Array(withCentroid.length), 1))

    centroids = new THREE.Points(geometry, makePointMaterial(renderer.getPixelRatio()))
    centroids.frustumCulled = false
    centroids.userData.classes = withCentroid
    scene.add(centroids)
  }

  /* ── Styling ──────────────────────────────────────────────────────────── */

  /** Recolour and resize every point from the current highlights and options. */
  function applyStyling() {
    if (!bundle || !points) return

    const { count } = bundle.manifest
    const color = points.geometry.getAttribute('color')
    const size = points.geometry.getAttribute('size')
    const alpha = points.geometry.getAttribute('alpha')
    const classesOf = bundle.items.classes
    const distances = bundle.items.distance
    const threshold = bundle.manifest.outlierThreshold

    const neutral = toRgbArray(NEUTRAL)
    const anyHighlight = highlights.size > 0
    const maxDistance = threshold > 0 ? threshold * 1.35 : 1

    for (let i = 0; i < count; i++) {
      let rgb = neutral
      let pointSize = BASE_POINT_SIZE
      let opacity = anyHighlight || options.outlierMode ? DIMMED_ALPHA : 0.9

      if (options.outlierMode) {
        const d = distances[i]
        if (d === null) {
          opacity = 0.25
        } else {
          rgb = outlierColor(d / maxDistance)
          const isOutlier = d >= threshold
          pointSize = isOutlier ? HIGHLIGHT_POINT_SIZE : BASE_POINT_SIZE
          opacity = isOutlier ? 1 : 0.55
        }
      } else if (anyHighlight) {
        // First matching selection wins, so a multi-label point takes the
        // colour of the class the user picked earliest — stable, not arbitrary.
        for (const [classId, hex] of highlights) {
          if (classesOf[i].includes(classId)) {
            rgb = toRgbArray(hex)
            pointSize = HIGHLIGHT_POINT_SIZE
            opacity = 1
            break
          }
        }
      }

      if (i === hovered) {
        pointSize = HIGHLIGHT_POINT_SIZE * 1.6
        opacity = 1
      }

      color.setXYZ(i, rgb[0], rgb[1], rgb[2])
      size.setX(i, pointSize)
      alpha.setX(i, opacity)
    }

    color.needsUpdate = true
    size.needsUpdate = true
    alpha.needsUpdate = true

    styleCentroids()
    rebuildHulls()
    rebuildAxes()
    invalidate()
  }

  function styleCentroids() {
    if (!centroids) return
    const entries = centroids.userData.classes
    const color = centroids.geometry.getAttribute('color')
    const size = centroids.geometry.getAttribute('size')
    const alpha = centroids.geometry.getAttribute('alpha')
    const neutral = toRgbArray(CENTROID_NEUTRAL)
    const show = options.showCentroids && !options.outlierMode

    entries.forEach((entry, i) => {
      const hex = highlights.get(entry.id)
      const rgb = hex ? toRgbArray(hex) : neutral
      color.setXYZ(i, rgb[0], rgb[1], rgb[2])
      size.setX(i, hex ? 13 : 7)
      // When a selection is active, unselected centroids would be visual noise.
      alpha.setX(i, !show ? 0 : hex ? 1 : highlights.size ? 0.25 : 0.8)
    })

    color.needsUpdate = true
    size.needsUpdate = true
    alpha.needsUpdate = true
  }

  function rebuildHulls() {
    clearGroup(hullGroup)
    if (!options.showHulls || options.outlierMode) return

    for (const [classId, hex] of highlights) {
      const entry = bundle.manifest.classes[classId]
      if (!entry?.hull) continue

      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array(entry.hull.vertices), 3),
      )
      geometry.setIndex(entry.hull.faces)
      geometry.computeVertexNormals()

      hullGroup.add(
        new THREE.Mesh(
          geometry,
          new THREE.MeshBasicMaterial({
            color: new THREE.Color(hex),
            transparent: true,
            opacity: 0.14,
            // Hull winding from qhull isn't consistently outward, and a
            // translucent shell reads better lit from both faces anyway.
            side: THREE.DoubleSide,
            depthWrite: false,
          }),
        ),
      )
    }
  }

  function rebuildAxes() {
    clearGroup(axisGroup)
    if (!options.showAxes || options.outlierMode) return

    for (const [classId, hex] of highlights) {
      const entry = bundle.manifest.classes[classId]
      if (!entry?.axis || !entry.centroid) continue

      const centre = new THREE.Vector3(...entry.centroid)
      const direction = new THREE.Vector3(...entry.axis)
      const [low, high] = entry.extent

      const geometry = new THREE.BufferGeometry().setFromPoints([
        centre.clone().addScaledVector(direction, low),
        centre.clone().addScaledVector(direction, high),
      ])

      axisGroup.add(
        new THREE.Line(
          geometry,
          new THREE.LineBasicMaterial({
            color: new THREE.Color(hex),
            transparent: true,
            opacity: 0.85,
          }),
        ),
      )
    }
  }

  /* ── Interaction ──────────────────────────────────────────────────────── */

  function pick() {
    if (!points || !pointerActive) return -1
    raycaster.setFromCamera(pointer, camera)
    const hits = raycaster.intersectObject(points, false)
    if (!hits.length) return -1

    // Nearest to the camera among everything under the cursor; the ray can
    // clip several points in a dense cluster.
    let best = hits[0]
    for (const hit of hits) if (hit.distance < best.distance) best = hit
    return best.index ?? -1
  }

  function updateHover() {
    const next = pick()
    if (next === hovered) return
    hovered = next
    applyStyling()
    onHover(hovered)
  }

  function emitLabels() {
    if (!centroids || !options.showCentroids || options.outlierMode) return onLabels([])

    const rect = canvas.getBoundingClientRect()
    const labels = []
    const vector = new THREE.Vector3()

    for (const entry of centroids.userData.classes) {
      if (!highlights.has(entry.id)) continue
      vector.set(...entry.centroid).project(camera)
      if (vector.z > 1) continue // behind the camera
      labels.push({
        id: entry.id,
        name: entry.name,
        color: highlights.get(entry.id),
        x: ((vector.x + 1) / 2) * rect.width,
        y: ((1 - vector.y) / 2) * rect.height,
      })
    }
    onLabels(labels)
  }

  /* ── Loop ─────────────────────────────────────────────────────────────── */

  function resize() {
    const rect = canvas.getBoundingClientRect()
    if (!rect.width || !rect.height) return
    renderer.setSize(rect.width, rect.height, false)
    camera.aspect = rect.width / rect.height
    camera.updateProjectionMatrix()
    invalidate()
  }

  function tick() {
    frame = requestAnimationFrame(tick)
    if (!running) return

    controls.update()
    if (controls.autoRotate) invalidate()
    if (!dirty) return

    updateHover()
    renderer.render(scene, camera)
    emitLabels()
    dirty = false
  }

  const observer = new ResizeObserver(resize)
  observer.observe(canvas)
  resize()
  tick()

  /* ── Teardown ─────────────────────────────────────────────────────────── */

  function clearGroup(group) {
    for (const child of [...group.children]) {
      group.remove(child)
      child.geometry?.dispose()
      child.material?.dispose()
    }
  }

  function disposePoints() {
    for (const object of [points, centroids]) {
      if (!object) continue
      scene.remove(object)
      object.geometry.dispose()
      object.material.dispose()
    }
    points = null
    centroids = null
    hovered = -1
  }

  function resetCamera() {
    camera.position.set(1.55, 1.02, 1.55)
    controls.target.set(0, 0, 0)
    controls.update()
    invalidate()
  }

  return {
    setBundle,
    setHighlights(next) {
      highlights = next
      applyStyling()
    },
    setOptions(next) {
      options = next
      controls.autoRotate = Boolean(next.spinning)
      applyStyling()
    },
    setPointer(event) {
      const rect = canvas.getBoundingClientRect()
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
      pointerActive = true
      invalidate()
    },
    clearPointer() {
      pointerActive = false
      invalidate()
    },
    /** Frame one class: centre on its centroid and pull the camera in. */
    focusClass(classId) {
      const entry = bundle?.manifest.classes[classId]
      if (!entry?.centroid) return
      controls.target.set(...entry.centroid)
      const offset = new THREE.Vector3(1.05, 0.7, 1.05)
      camera.position.copy(controls.target).add(offset)
      controls.update()
      invalidate()
    },
    resetCamera,
    getHovered: () => hovered,
    onHover(cb) {
      onHover = cb
    },
    onLabels(cb) {
      onLabels = cb
    },
    /** Stops the loop entirely when the canvas scrolls out of view. */
    setRunning(next) {
      running = next
      if (next) invalidate()
    },
    dispose() {
      cancelAnimationFrame(frame)
      observer.disconnect()
      controls.removeEventListener('change', invalidate)
      controls.dispose()
      clearGroup(hullGroup)
      clearGroup(axisGroup)
      scene.remove(originAxes)
      originAxes.geometry.dispose()
      originAxes.material.dispose()
      disposePoints()
      renderer.dispose()
    },
  }
}
