/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package app.pivo.android.basicsdkdemo;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import app.pivo.android.basicsdk.PivoSdk;
import app.pivo.android.basicsdkdemo.customview.OverlayView;
import app.pivo.android.basicsdkdemo.customview.OverlayView.DrawCallback;
import app.pivo.android.basicsdkdemo.env.BorderedText;
import app.pivo.android.basicsdkdemo.env.ImageUtils;
import app.pivo.android.basicsdkdemo.env.Logger;
import app.pivo.android.basicsdkdemo.tflite.Classifier;
import app.pivo.android.basicsdkdemo.tflite.YoloV4Classifier;
import app.pivo.android.basicsdkdemo.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener, View.OnClickListener {
    private static final Logger LOGGER = new Logger();

    private static int TF_OD_API_INPUT_SIZE = 320;
    private static int[] TF_OD_API_OUTPUT_SHAPE = {1500, 1500};
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "yolov4-tiny-320.tflite";

    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/obj.names";

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(1920 , 1080);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private Classifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector =
                    YoloV4Classifier.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_IS_QUANTIZED,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_OUTPUT_SHAPE,
                            true);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    private float previousCenter = previewWidth / 2;

    @Override
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);

        findViewById(R.id.select_person).setOnClickListener(this);
        findViewById(R.id.select_dog).setOnClickListener(this);
        findViewById(R.id.select_cat).setOnClickListener(this);
        findViewById(R.id.select_all).setOnClickListener(this);

        findViewById(R.id.model_performance).setOnClickListener(this);
        findViewById(R.id.model_speed).setOnClickListener(this);
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        List<Classifier.Recognition> temp = detector.recognizeImage(croppedBitmap);
                        List<Classifier.Recognition> results;

                        if (selectedObject.equals("all")) {
                            results = temp;
                        } else {
                            results = new ArrayList<>();

                            for (Classifier.Recognition record: temp) {
                                if (record.getTitle().equals(selectedObject)) results.add(record);
                            }
                        }
                        if (results.size() > 0) {
                            float position = results.get(0).getLocation().centerX();

                            runInBackground(() -> {
                                if (position < TF_OD_API_INPUT_SIZE / 2 - 10) {
                                    PivoSdk.getInstance().turnLeft(1);
                                } else if (TF_OD_API_INPUT_SIZE / 2 + 10 < position) {
                                    PivoSdk.getInstance().turnRight(1);
                                } else {
                                    PivoSdk.getInstance().stop();
                                }
                            });

                            runOnUiThread(() -> {

                            });

                            Log.i("Position", "" + position);
                        }

                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        Log.e("CHECK", "run: " + results.size());

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();

                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {
                                canvas.drawRect(location, paint);

                                cropToFrameTransform.mapRect(location);

                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }
                        }

                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        trackingOverlay.postInvalidate();

                        computingDetection = false;

                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showFrameInfo(previewWidth + "x" + previewHeight);
                                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });
                    }
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }

    private String model_name = "performance";
    private String modelSelect = TF_OD_API_MODEL_FILE;
    private int[] output_shape = TF_OD_API_OUTPUT_SHAPE;
    private int input_size = TF_OD_API_INPUT_SIZE;

    @Override
    public void onClick(View v) {
        super.onClick(v);

        switch(v.getId()) {
            case R.id.model_performance:
                model_name = "performance";
                modelSelect = TF_OD_API_MODEL_FILE;
                output_shape = TF_OD_API_OUTPUT_SHAPE;
                input_size = TF_OD_API_INPUT_SIZE;
                break;

            case R.id.model_speed:
                model_name = "speed";
                modelSelect = "yolov4-tiny-320-lite.tflite";
                output_shape = new int[]{ 1500, 1500 };
                input_size = 320;
                break;

            case R.id.select_all:
                selectedObject = "all";
                break;

            case R.id.select_person:
                selectedObject = "person";
                break;

            case R.id.select_dog:
                selectedObject = "dog";
                break;

            case R.id.select_cat:
                selectedObject = "cat";
                break;
        }

        Classifier temp = detector;
        try {
            detector = YoloV4Classifier.create(
                    getAssets(),
                    modelSelect,
                    TF_OD_API_LABELS_FILE,
                    TF_OD_API_IS_QUANTIZED,
                    input_size,
                    output_shape,
                    true);

            TF_OD_API_INPUT_SIZE = input_size;
        } catch (IOException e) {
            e.printStackTrace();
            detector = temp;
        }

        ((TextView)findViewById(R.id.selected_object)).setText(selectedObject);
        ((TextView)findViewById(R.id.select_model)).setText(model_name);

    }
}
