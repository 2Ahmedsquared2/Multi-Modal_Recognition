import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import AudioInputZone from "../components/AudioInputZone";
import AudioTrimmer from "../components/AudioTrimmer";
import ImageInputZone from "../components/ImageInputZone";
import MicrophoneRecorder from "../components/MicrophoneRecorder";
import type { BrushTool } from "../components/SpectrogramEditor";
import SpectrogramEditor from "../components/SpectrogramEditor";
import { useModel } from "../contexts/ModelContext";
import type { WhatIfResponse } from "../types";

// ── Confidence bar list ──────────────────────────────────────────────────

function ConfidenceBars({
	prediction,
	originalConfidences,
}: {
	prediction: WhatIfResponse;
	originalConfidences?: Record<string, number>;
}) {
	const sorted = Object.entries(prediction.all_confidences).sort(
		([, a], [, b]) => b - a,
	);

	return (
		<div className="space-y-1.5">
			{sorted.slice(0, 5).map(([cls, prob], idx) => {
				const isTop = idx === 0;
				const origProb =
					originalConfidences !== undefined
						? (originalConfidences[cls] ?? 0)
						: null;
				const delta = origProb !== null ? prob - origProb : null;

				return (
					<div key={cls} className="space-y-0.5">
						<div className="flex justify-between items-center text-xs">
							<span
								className={
									isTop
										? "font-semibold text-warm-900 dark:text-warm-100 capitalize"
										: "text-warm-500 dark:text-warm-400 capitalize"
								}
							>
								{cls}
							</span>
							<div className="flex items-center gap-1.5">
								{delta !== null && Math.abs(delta) > 0.005 && (
									<span
										className={`text-[10px] font-mono ${
											delta > 0 ? "text-emerald-500" : "text-rose-500"
										}`}
									>
										{delta > 0 ? "+" : ""}
										{(delta * 100).toFixed(1)}
									</span>
								)}
								<span
									className={`font-mono ${
										isTop
											? "text-accent dark:text-accent-light font-semibold"
											: "text-warm-400 dark:text-warm-600"
									}`}
								>
									{(prob * 100).toFixed(1)}%
								</span>
							</div>
						</div>
						<div className="h-1.5 bg-warm-200 dark:bg-warm-700 rounded-full overflow-hidden">
							<div
								className={`h-full rounded-full transition-all duration-500 ease-out ${
									isTop
										? "bg-accent dark:bg-accent-light"
										: "bg-slate-300 dark:bg-slate-700"
								}`}
								style={{ width: `${prob * 100}%` }}
							/>
						</div>
					</div>
				);
			})}
		</div>
	);
}

// ── Preset definitions ───────────────────────────────────────────────────

const AUDIO_PRESETS = [
	{ id: "remove-low", label: "Remove Low Freq", icon: "⬇" },
	{ id: "remove-high", label: "Remove High Freq", icon: "⬆" },
	{ id: "keep-mid", label: "Keep Mid-Range", icon: "⬌" },
	{ id: "add-noise", label: "Add Noise", icon: "〰" },
	{ id: "reset", label: "Reset", icon: "↺" },
] as const;

const IMAGE_PRESETS = [
	{ id: "remove-top", label: "Blank Top Half", icon: "⬆" },
	{ id: "remove-bottom", label: "Blank Bottom Half", icon: "⬇" },
	{ id: "keep-center", label: "Keep Center", icon: "◎" },
	{ id: "add-noise", label: "Add Noise", icon: "〰" },
	{ id: "reset", label: "Reset", icon: "↺" },
] as const;

// ── Phase type ───────────────────────────────────────────────────────────

type Phase = "upload" | "trim" | "recording" | "loaded";

// ── Main page component ─────────────────────────────────────────────────

export default function WhatIf() {
	const { engine, engineLabel, dataset, modality, activeDataset } = useModel();

	const isAudio = modality === "audio";
	const datasetLabel = activeDataset?.name ?? dataset;

	// Phase & audio source
	const [phase, setPhase] = useState<Phase>("upload");
	const [audioFile, setAudioFile] = useState<File | null>(null);

	// Spectrogram / preprocessed matrix & predictions
	const [originalSpec, setOriginalSpec] = useState<number[][] | null>(null);
	const [modifiedSpec, setModifiedSpec] = useState<number[][] | null>(null);
	const [originalPred, setOriginalPred] = useState<WhatIfResponse | null>(null);
	const [modifiedPred, setModifiedPred] = useState<WhatIfResponse | null>(null);

	// Editor state
	const [tool, setTool] = useState<BrushTool>("paint");
	const [brushSize, setBrushSize] = useState(3);
	const [loading, setLoading] = useState(false);
	const [predicting, setPredicting] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [insight, setInsight] = useState("");

	const originalPredRef = useRef<WhatIfResponse | null>(null);

	// Reset when modality/dataset changes
	useEffect(() => {
		resetState();
	}, [modality, dataset]);

	// ── Shared: Reset everything ──

	const resetState = useCallback(() => {
		setPhase("upload");
		setAudioFile(null);
		setOriginalSpec(null);
		setModifiedSpec(null);
		setOriginalPred(null);
		setModifiedPred(null);
		originalPredRef.current = null;
		setError(null);
		setInsight("");
	}, []);

	// ──────────────────────────────────────────────────────────────────────
	//  Audio flow
	// ──────────────────────────────────────────────────────────────────────

	const handleAudioFileSelected = useCallback((file: File) => {
		setError(null);
		setAudioFile(file);
		setPhase("trim");
	}, []);

	const handleMicRecording = useCallback((file: File) => {
		setError(null);
		setAudioFile(file);
		setPhase("trim");
	}, []);

	const handleClassifyBlob = useCallback(
		async (blob: Blob) => {
			setError(null);
			setLoading(true);
			try {
				const wavFile = new File([blob], "segment.wav", { type: "audio/wav" });
				const result = await api.classify(wavFile, engine, dataset);
				const spec = result.spectrogram;
				const pred: WhatIfResponse = {
					prediction: result.prediction,
					confidence: result.confidence,
					all_confidences: result.all_confidences,
				};

				setOriginalSpec(spec);
				setModifiedSpec(spec.map((row) => [...row]));
				setOriginalPred(pred);
				setModifiedPred(pred);
				originalPredRef.current = pred;
				setInsight("");
				setPhase("loaded");
			} catch (err) {
				setError(
					err instanceof Error ? err.message : "Failed to process audio",
				);
			} finally {
				setLoading(false);
			}
		},
		[engine, dataset],
	);

	// ──────────────────────────────────────────────────────────────────────
	//  Image flow
	// ──────────────────────────────────────────────────────────────────────

	const handleImageFileSelected = useCallback(
		async (file: File) => {
			setError(null);
			setLoading(true);
			try {
				const result = await api.classify(file, engine, dataset);
				const spec = result.spectrogram; // preprocessed image matrix
				const pred: WhatIfResponse = {
					prediction: result.prediction,
					confidence: result.confidence,
					all_confidences: result.all_confidences,
				};

				setOriginalSpec(spec);
				setModifiedSpec(spec.map((row) => [...row]));
				setOriginalPred(pred);
				setModifiedPred(pred);
				originalPredRef.current = pred;
				setInsight("");
				setPhase("loaded");
			} catch (err) {
				setError(
					err instanceof Error ? err.message : "Failed to process image",
				);
				setPhase("upload");
			} finally {
				setLoading(false);
			}
		},
		[engine, dataset],
	);

	// ──────────────────────────────────────────────────────────────────────
	//  Shared: Re-predict modified input
	// ──────────────────────────────────────────────────────────────────────

	const predict = useCallback(
		async (spec: number[][]) => {
			setPredicting(true);
			try {
				const result = await api.whatIf(spec, engine, dataset);
				setModifiedPred(result);

				// Generate insight by comparing to original
				const orig = originalPredRef.current;
				if (orig) {
					if (result.prediction !== orig.prediction) {
						setInsight(
							`The prediction shifted from ${orig.prediction} ` +
								`(${(orig.confidence * 100).toFixed(1)}%) to ` +
								`${result.prediction} (${(result.confidence * 100).toFixed(1)}%). ` +
								`This suggests the model relies on the modified regions ` +
								`to distinguish these classes.`,
						);
					} else if (result.confidence < orig.confidence - 0.1) {
						setInsight(
							`Still ${result.prediction}, but confidence dropped from ` +
								`${(orig.confidence * 100).toFixed(1)}% to ` +
								`${(result.confidence * 100).toFixed(1)}%. ` +
								`The edited regions contribute to the model's certainty.`,
						);
					} else if (result.confidence > orig.confidence + 0.05) {
						setInsight(
							`Confidence increased from ` +
								`${(orig.confidence * 100).toFixed(1)}% to ` +
								`${(result.confidence * 100).toFixed(1)}%. ` +
								`Your edits reinforced the model's decision.`,
						);
					} else {
						setInsight(
							`No significant change — the model's prediction is robust to these modifications.`,
						);
					}
				}
			} catch (err) {
				console.error("What-if prediction failed:", err);
			} finally {
				setPredicting(false);
			}
		},
		[engine, dataset],
	);

	// ── Canvas edit handler ──

	const handleSpecChange = useCallback(
		(newSpec: number[][]) => {
			setModifiedSpec(newSpec);
			predict(newSpec);
		},
		[predict],
	);

	// ── Preset experiments ──

	const applyPreset = useCallback(
		(presetId: string) => {
			if (!originalSpec) return;
			const spec = originalSpec.map((row) => [...row]);
			const nRows = spec.length;
			const nCols = spec[0]?.length ?? 0;

			switch (presetId) {
				// Audio presets
				case "remove-low": {
					const cutoff = Math.floor(nRows / 3);
					for (let r = 0; r < cutoff; r++)
						for (let c = 0; c < nCols; c++) spec[r][c] = 0;
					break;
				}
				case "remove-high": {
					const cutoff = Math.floor((nRows * 2) / 3);
					for (let r = cutoff; r < nRows; r++)
						for (let c = 0; c < nCols; c++) spec[r][c] = 0;
					break;
				}
				case "keep-mid": {
					const lo = Math.floor(nRows / 3);
					const hi = Math.floor((nRows * 2) / 3);
					for (let r = 0; r < nRows; r++) {
						if (r < lo || r >= hi) {
							for (let c = 0; c < nCols; c++) spec[r][c] = 0;
						}
					}
					break;
				}
				// Image presets
				case "remove-top": {
					const cutoff = Math.floor(nRows / 2);
					for (let r = 0; r < cutoff; r++)
						for (let c = 0; c < nCols; c++) spec[r][c] = 0;
					break;
				}
				case "remove-bottom": {
					const cutoff = Math.floor(nRows / 2);
					for (let r = cutoff; r < nRows; r++)
						for (let c = 0; c < nCols; c++) spec[r][c] = 0;
					break;
				}
				case "keep-center": {
					const marginR = Math.floor(nRows / 4);
					const marginC = Math.floor(nCols / 4);
					for (let r = 0; r < nRows; r++)
						for (let c = 0; c < nCols; c++)
							if (
								r < marginR ||
								r >= nRows - marginR ||
								c < marginC ||
								c >= nCols - marginC
							)
								spec[r][c] = 0;
					break;
				}
				// Shared presets
				case "add-noise": {
					for (let r = 0; r < nRows; r++)
						for (let c = 0; c < nCols; c++)
							spec[r][c] = Math.max(
								0,
								Math.min(1, spec[r][c] + (Math.random() - 0.5) * 0.3),
							);
					break;
				}
				case "reset":
					// spec is already a fresh copy of originalSpec
					break;
			}

			setModifiedSpec(spec);
			predict(spec);
		},
		[originalSpec, predict],
	);

	const presets = isAudio ? AUDIO_PRESETS : IMAGE_PRESETS;
	const inputLabel = isAudio ? "spectrogram" : "preprocessed image";

	return (
		<div className="max-w-6xl mx-auto px-4 sm:px-8 py-10 space-y-8">
			{/* ── Header ── */}
			<header className="stagger-1 space-y-2">
				<div className="flex items-center gap-2 flex-wrap">
					<h1 className="font-display text-2xl font-bold tracking-tight text-warm-900 dark:text-warm-100">
						What-If Tool
					</h1>
					<span className="badge-accent">Interactive</span>
					<span
						className={`text-[10px] font-medium px-2 py-0.5 rounded-full
            ${
							engine === "custom"
								? "bg-accent/15 text-accent dark:bg-accent/10 dark:text-accent-light"
								: "bg-warm-200 text-warm-600 dark:bg-warm-700 dark:text-warm-400"
						}`}
					>
						{engineLabel}
					</span>
					<span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-warm-200 text-warm-600 dark:bg-warm-700 dark:text-warm-400">
						{datasetLabel}
					</span>
				</div>
				<p className="text-sm text-warm-600 dark:text-warm-400">
					{isAudio
						? `Paint on a spectrogram and watch the ${engineLabel} model's prediction change in real time. Discover which frequency regions matter most for each class.`
						: `Paint on the preprocessed image and watch the ${engineLabel} model's prediction change in real time. Discover which regions matter most for classification.`}
				</p>
			</header>

			{/* ═══════════════════════════════════════════════════════════════════
          AUDIO FLOW
          ═══════════════════════════════════════════════════════════════════ */}
			{isAudio && (
				<>
					{phase === "upload" && (
						<AudioInputZone
							onFileSelected={handleAudioFileSelected}
							onRecordClick={() => setPhase("recording")}
							statusHint="Upload or record audio, then select a 2-second segment to paint on and explore."
						/>
					)}

					{phase === "recording" && (
						<MicrophoneRecorder
							onRecordingComplete={handleMicRecording}
							onCancel={resetState}
						/>
					)}

					{phase === "trim" && audioFile && (
						<div className="space-y-4">
							<AudioTrimmer
								file={audioFile}
								clipDuration={2}
								onClassify={handleClassifyBlob}
								onCancel={resetState}
							/>

							{loading && (
								<div className="card p-6 flex items-center gap-3 animate-fade-in">
									<div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
									<div>
										<p className="text-sm font-medium text-warm-700 dark:text-warm-300">
											Analyzing audio…
										</p>
										<p className="text-xs text-warm-500 dark:text-warm-500">
											Generating spectrogram → Running inference
										</p>
									</div>
								</div>
							)}

							{error && (
								<div className="card p-4 border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/5 animate-fade-in">
									<p className="text-sm text-rose-600 dark:text-rose-400">
										{error}
									</p>
								</div>
							)}
						</div>
					)}
				</>
			)}

			{/* ═══════════════════════════════════════════════════════════════════
          IMAGE FLOW
          ═══════════════════════════════════════════════════════════════════ */}
			{!isAudio && (
				<>
					{phase === "upload" && !loading && (
						<ImageInputZone
							onFileSelected={handleImageFileSelected}
							statusHint={`Upload an image for ${datasetLabel} — then paint on the preprocessed input to explore.`}
						/>
					)}

					{loading && phase === "upload" && (
						<div className="card p-8 flex flex-col items-center gap-4 animate-fade-in">
							<div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
							<div className="text-center">
								<p className="text-sm font-medium text-warm-700 dark:text-warm-300">
									Processing image…
								</p>
								<p className="text-xs text-warm-500 dark:text-warm-500 mt-1">
									Resizing → Normalizing → Running inference
								</p>
							</div>
						</div>
					)}

					{error && phase === "upload" && (
						<div className="card p-4 border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/5 animate-fade-in">
							<p className="text-sm text-rose-600 dark:text-rose-400">
								{error}
							</p>
						</div>
					)}
				</>
			)}

			{/* ═══════════════════════════════════════════════════════════════════
          LOADED — Shared editor (works on spectrogram or image matrix)
          ═══════════════════════════════════════════════════════════════════ */}
			{phase === "loaded" && originalSpec && modifiedSpec && (
				<div className="space-y-6">
					{/* ── Top bar: source controls + predicting indicator ── */}
					<div className="flex items-center justify-between">
						<div className="flex items-center gap-2">
							{isAudio && (
								<button
									onClick={() => {
										setPhase("trim");
									}}
									className="px-3 py-1.5 rounded-lg text-xs font-medium
                    bg-accent-subtle text-accent dark:bg-accent/10 dark:text-accent-light
                    hover:bg-accent/15 dark:hover:bg-accent/20
                    transition-all duration-150 active:scale-[0.98]"
								>
									Try Different Segment
								</button>
							)}
							<button
								onClick={resetState}
								className="px-3 py-1.5 rounded-lg text-xs font-medium
                  bg-warm-200 text-warm-600 dark:bg-warm-700 dark:text-warm-400
                  hover:bg-warm-300 dark:hover:bg-warm-600
                  transition-all duration-150 active:scale-[0.98]"
							>
								{isAudio ? "New Audio" : "New Image"}
							</button>
						</div>
						{predicting && (
							<div className="flex items-center gap-1.5 text-xs text-accent">
								<div className="w-3 h-3 border-2 border-accent border-t-transparent rounded-full animate-spin" />
								Predicting…
							</div>
						)}
					</div>

					{/* ── Before / After ── */}
					<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
						{/* Original (read-only) */}
						<div className="space-y-4">
							<div className="card p-4">
								<SpectrogramEditor
									spectrogram={originalSpec}
									readOnly
									label={`Original ${inputLabel}`}
								/>
							</div>
							<div className="card p-4 space-y-3">
								<div className="flex items-baseline gap-2">
									<span className="text-sm font-bold text-warm-900 dark:text-warm-100 capitalize">
										{originalPred?.prediction ?? "—"}
									</span>
									<span className="text-xs font-mono text-accent">
										{originalPred
											? `${(originalPred.confidence * 100).toFixed(1)}%`
											: ""}
									</span>
								</div>
								{originalPred && <ConfidenceBars prediction={originalPred} />}
							</div>
						</div>

						{/* Modified (editable) */}
						<div className="space-y-4">
							<div className="card p-4">
								<SpectrogramEditor
									spectrogram={modifiedSpec}
									onChange={handleSpecChange}
									tool={tool}
									brushSize={brushSize}
									label="Modified — draw here"
								/>
							</div>
							<div className="card p-4 space-y-3">
								<div className="flex items-baseline gap-2">
									<span
										className={`text-sm font-bold capitalize ${
											modifiedPred?.prediction !== originalPred?.prediction
												? "text-amber-500"
												: "text-warm-900 dark:text-warm-100"
										}`}
									>
										{modifiedPred?.prediction ?? "—"}
									</span>
									<span className="text-xs font-mono text-accent">
										{modifiedPred
											? `${(modifiedPred.confidence * 100).toFixed(1)}%`
											: ""}
									</span>
									{modifiedPred &&
										originalPred &&
										modifiedPred.prediction !== originalPred.prediction && (
											<span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400 font-medium">
												Changed
											</span>
										)}
								</div>
								{modifiedPred && originalPred && (
									<ConfidenceBars
										prediction={modifiedPred}
										originalConfidences={originalPred.all_confidences}
									/>
								)}
							</div>
						</div>
					</div>

					{/* ── Tool palette ── */}
					<div className="card p-4">
						<div className="flex flex-wrap items-center gap-6">
							{/* Brush tool */}
							<div className="flex items-center gap-2">
								<span className="text-[11px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500 mr-1">
									Tool
								</span>
								<button
									onClick={() => setTool("paint")}
									className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
										tool === "paint"
											? "bg-amber-100 text-amber-700 dark:bg-amber-500/15 dark:text-amber-400"
											: "bg-warm-200 text-warm-500 dark:bg-warm-700 dark:text-warm-400 hover:bg-warm-300 dark:hover:bg-warm-600"
									}`}
								>
									Paint (+)
								</button>
								<button
									onClick={() => setTool("erase")}
									className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
										tool === "erase"
											? "bg-rose-100 text-rose-700 dark:bg-rose-500/15 dark:text-rose-400"
											: "bg-warm-200 text-warm-500 dark:bg-warm-700 dark:text-warm-400 hover:bg-warm-300 dark:hover:bg-warm-600"
									}`}
								>
									Erase (−)
								</button>
							</div>

							{/* Brush size */}
							<div className="flex items-center gap-2">
								<span className="text-[11px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500">
									Size
								</span>
								{([1, 3, 6] as const).map((size) => (
									<button
										key={size}
										onClick={() => setBrushSize(size)}
										className={`w-8 h-8 rounded-lg text-xs font-medium transition-colors
                      flex items-center justify-center ${
												brushSize === size
													? "bg-accent/15 text-accent dark:bg-accent/15 dark:text-accent-light"
													: "bg-warm-200 text-warm-500 dark:bg-warm-700 dark:text-warm-400 hover:bg-warm-300 dark:hover:bg-warm-600"
											}`}
									>
										{size === 1 ? "S" : size === 3 ? "M" : "L"}
									</button>
								))}
							</div>

							{/* Divider */}
							<div className="w-px h-6 bg-warm-300 dark:bg-warm-600" />

							{/* Presets */}
							<div className="flex items-center gap-2 flex-wrap">
								<span className="text-[11px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500 mr-1">
									Presets
								</span>
								{presets.map((p) => (
									<button
										key={p.id}
										onClick={() => applyPreset(p.id)}
										className={`px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${
											p.id === "reset"
												? "bg-warm-300 text-warm-700 dark:bg-warm-600 dark:text-warm-300 hover:bg-warm-400 dark:hover:bg-warm-500"
												: "bg-warm-200 text-warm-500 dark:bg-warm-700 dark:text-warm-400 hover:bg-warm-300 dark:hover:bg-warm-600"
										}`}
									>
										<span className="mr-1">{p.icon}</span>
										{p.label}
									</button>
								))}
							</div>
						</div>
					</div>

					{/* ── Insight panel ── */}
					{insight && (
						<div className="card p-4 border-accent/20 dark:border-accent/20 bg-accent-subtle dark:bg-accent/5">
							<div className="flex gap-3">
								<div className="shrink-0 w-8 h-8 rounded-lg bg-accent-subtle dark:bg-accent/10 flex items-center justify-center">
									<svg
										className="w-4 h-4 text-accent"
										viewBox="0 0 24 24"
										fill="none"
										stroke="currentColor"
										strokeWidth={2}
										strokeLinecap="round"
										strokeLinejoin="round"
									>
										<circle cx="12" cy="12" r="10" />
										<path d="M12 16v-4M12 8h.01" />
									</svg>
								</div>
								<div>
									<p className="text-[11px] font-semibold uppercase tracking-wider text-accent dark:text-accent-light mb-1">
										Insight
									</p>
									<p className="text-sm text-warm-700 dark:text-warm-300">
										{insight}
									</p>
								</div>
							</div>
						</div>
					)}
				</div>
			)}
		</div>
	);
}
