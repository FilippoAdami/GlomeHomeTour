plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.glomehometour.capture"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.glomehometour.capture"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"

        // Redmi Note 10S (and every realistic target device) is arm64 — skip the other
        // ABIs ONNX Runtime ships to keep the APK small. Drop this filter if x86_64 emulator
        // testing is ever needed.
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    // ZipDepth ONNX export (~25MB) is a real binary asset, not source — don't compress twice.
    androidResources {
        noCompress += "onnx"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")

    // ARCore now owns the camera (see ArCoreCameraSource) — it needs exclusive Camera2 access
    // for VIO, so CameraX (which also opens Camera2 directly) is dropped rather than shared.
    implementation("com.google.ar:core:1.54.0")

    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")
}
