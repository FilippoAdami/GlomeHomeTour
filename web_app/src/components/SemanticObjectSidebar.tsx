// src/components/SemanticObjectSidebar.tsx
import { useState } from 'react';
import { SemanticCategory, ObjectInstance } from '../lib/api';

interface Props {
  categories: SemanticCategory[];
  selectedCategory: string | null;
  onSelectCategory: (categoryId: string | null) => void;
  onFocusInstance?: (instance: ObjectInstance) => void;
}

export function SemanticObjectSidebar({
  categories,
  selectedCategory,
  onSelectCategory,
  onFocusInstance,
}: Props) {
  const [isOpen, setIsOpen] = useState(true);

  const totalObjects = categories.reduce((acc, c) => acc + c.count, 0);

  return (
    <div
      className={`absolute top-20 left-5 z-20 flex flex-col bg-slate-900/90 backdrop-blur-md border border-slate-700/60 rounded-2xl shadow-2xl transition-all duration-300 max-h-[calc(100vh-140px)] ${
        isOpen ? 'w-72' : 'w-12'
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-3.5 border-b border-slate-800">
        {isOpen ? (
          <div className="flex items-center gap-2">
            <span className="text-base">🏷️</span>
            <div>
              <h3 className="text-xs font-bold text-white tracking-wide uppercase">
                Scene Inventory
              </h3>
              <p className="text-[11px] text-slate-400">
                {totalObjects} physical instances detected
              </p>
            </div>
          </div>
        ) : (
          <span className="text-base mx-auto">🏷️</span>
        )}

        <button
          onClick={() => setIsOpen(!isOpen)}
          className="text-slate-400 hover:text-white text-xs p-1 rounded-md transition-colors"
          title={isOpen ? 'Collapse panel' : 'Expand panel'}
        >
          {isOpen ? '◀' : '▶'}
        </button>
      </div>

      {/* Content list */}
      {isOpen && (
        <div className="p-3 space-y-2 overflow-y-auto flex-1 custom-scrollbar">
          {categories.map((cat) => {
            const isSelected = selectedCategory === cat.id;

            return (
              <div
                key={cat.id}
                className={`rounded-xl border transition-all cursor-pointer ${
                  isSelected
                    ? 'border-blue-500 bg-blue-950/40 shadow-sm'
                    : 'border-slate-800/80 bg-slate-950/40 hover:border-slate-700 hover:bg-slate-800/40'
                }`}
                onClick={() => onSelectCategory(isSelected ? null : cat.id)}
              >
                <div className="flex items-center justify-between p-2.5">
                  <div className="flex items-center gap-2.5">
                    <span className="text-lg">{cat.icon}</span>
                    <div>
                      <p className="text-xs font-semibold text-white">{cat.label}</p>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        <span
                          className="w-2 h-2 rounded-full inline-block"
                          style={{ backgroundColor: cat.color }}
                        />
                        <span className="text-[10px] text-slate-400">
                          {cat.count} {cat.count === 1 ? 'item' : 'items'}
                        </span>
                      </div>
                    </div>
                  </div>

                  <span
                    className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                      isSelected
                        ? 'bg-blue-500 text-white'
                        : 'bg-slate-800 text-slate-300'
                    }`}
                  >
                    {isSelected ? 'Active' : 'Show'}
                  </span>
                </div>

                {/* Expanded instances breakdown */}
                {isSelected && cat.instances.length > 0 && (
                  <div className="px-2.5 pb-2.5 pt-1 border-t border-slate-800/80 space-y-1.5">
                    {cat.instances.map((inst) => (
                      <div
                        key={inst.id}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (onFocusInstance) onFocusInstance(inst);
                        }}
                        className="flex items-center justify-between p-1.5 rounded-lg bg-slate-900/60 hover:bg-slate-800/60 transition-colors text-[11px]"
                      >
                        <span className="text-slate-200 font-medium">{inst.name}</span>
                        {inst.dimensions && (
                          <span className="text-[10px] text-slate-400">
                            {inst.dimensions}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
