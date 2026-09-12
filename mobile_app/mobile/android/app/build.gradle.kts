plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.glomehometour.arscan"
    compileSdk = 34

    defaultConfig {
        // API 30 per SPEC section 3; also what lets DatasetWriter use MediaStore under
        // Documents/ with no storage permission and no legacy fallback path.
        applicationId = "com.glomehometour.arscan"
        minSdk = 30
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"

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
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.ar:core:1.54.0")
    // On-device monocular depth, guidance only: the target device has no ARCore Depth API.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

    testImplementation("junit:junit:4.13.2")
}
