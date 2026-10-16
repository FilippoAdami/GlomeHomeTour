declare module '@mkkellogg/gaussian-splats-3d' {
  export class Viewer {
    constructor(options: { rootElement: HTMLElement; sharedMemoryForWorkers: boolean; ignoreDevicePixelRatio: boolean });
    addSplatScene(path: string, options?: {
      showLoadingUI?: boolean;
      progressiveLoad?: boolean;
      position?: [number, number, number];
      rotation?: [number, number, number, number];
    }): Promise<unknown>;
    start(): void;
    dispose(): Promise<void>;
  }
}
