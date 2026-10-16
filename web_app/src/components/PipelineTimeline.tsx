// src/components/PipelineTimeline.tsx
import { PipelineJob, PipelineStage } from '../lib/api';

interface Props {
  job: PipelineJob | null;
  overallStatus: string;
}

export function PipelineTimeline({ job, overallStatus }: Props) {
  if (!job || !job.stages || job.stages.length === 0) {
    return (
      <div className="card p-8">
        <h2 className="text-base font-semibold text-glome-navy mb-2">Reconstruction Pipeline</h2>
        <p className="text-sm text-gray-400">
          {overallStatus === 'uploading'
            ? 'Archive is uploading to local server…'
            : 'No pipeline job data recorded yet.'}
        </p>
      </div>
    );
  }

  const stages = job.stages;
  const totalTime = stages.reduce((acc, s) => acc + (s.elapsedSeconds || 0), 0);

  return (
    <div className="card p-8 space-y-6">
      <div className="flex items-center justify-between pb-4 border-b border-gray-100">
        <div>
          <h2 className="text-base font-bold text-glome-navy">Reconstruction Pipeline Stages</h2>
          <p className="text-xs text-gray-400 mt-0.5">
            AMD ROCm GPU Pipeline ({stages.filter((s) => s.status === 'done').length}/{stages.length} steps completed)
          </p>
        </div>

        {totalTime > 0 && (
          <div className="text-right">
            <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-blue-50 text-glome-blue border border-blue-100">
              ⏱ Total Time: {totalTime.toFixed(1)}s
            </span>
          </div>
        )}
      </div>

      <div className="space-y-4">
        {stages.map((stage: PipelineStage, idx: number) => {
          const isDone = stage.status === 'done';
          const isRunning = stage.status === 'running';
          const isError = stage.status === 'error';

          return (
            <div
              key={stage.name}
              className={`p-4 rounded-xl border transition-all ${
                isRunning
                  ? 'border-glome-blue/40 bg-blue-50/40 shadow-sm'
                  : isDone
                  ? 'border-gray-100 bg-white'
                  : 'border-dashed border-gray-200 bg-gray-50/50 opacity-70'
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-3.5">
                  {/* Status Indicator Icon */}
                  <div
                    className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 mt-0.5 ${
                      isDone
                        ? 'bg-emerald-500 text-white'
                        : isRunning
                        ? 'bg-glome-blue text-white animate-spin'
                        : isError
                        ? 'bg-red-500 text-white'
                        : 'bg-gray-200 text-gray-500'
                    }`}
                  >
                    {isDone ? '✓' : isRunning ? '⚙' : isError ? '✕' : idx + 1}
                  </div>

                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="text-sm font-semibold text-glome-navy">
                        {stage.displayName}
                      </h3>
                      {stage.frames && (
                        <span className="text-[11px] font-medium text-gray-400 bg-gray-100 px-2 py-0.5 rounded">
                          {stage.frames} frames
                        </span>
                      )}
                    </div>
                    {stage.description && (
                      <p className="text-xs text-gray-500 mt-1 leading-relaxed">
                        {stage.description}
                      </p>
                    )}
                  </div>
                </div>

                {/* Right timing / status badge */}
                <div className="shrink-0 text-right">
                  {isDone && stage.elapsedSeconds !== undefined ? (
                    <span className="text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-200/60 px-2.5 py-1 rounded-full">
                      ✓ {stage.elapsedSeconds.toFixed(2)}s
                    </span>
                  ) : isRunning ? (
                    <span className="text-xs font-medium text-glome-blue bg-blue-100/70 px-2.5 py-1 rounded-full animate-pulse">
                      Processing…
                    </span>
                  ) : isError ? (
                    <span className="text-xs font-medium text-red-700 bg-red-100 px-2.5 py-1 rounded-full">
                      Error
                    </span>
                  ) : (
                    <span className="text-xs font-medium text-gray-400">
                      Queued
                    </span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
