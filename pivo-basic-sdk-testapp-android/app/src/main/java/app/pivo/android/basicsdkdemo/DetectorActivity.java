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

import static org.opencv.android.Utils.bitmapToMat;
import static org.opencv.android.Utils.matToBitmap;

import android.content.Context;
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
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import app.pivo.android.basicsdk.PivoSdk;
import app.pivo.android.basicsdkdemo.customview.OverlayView;
import app.pivo.android.basicsdkdemo.customview.OverlayView.DrawCallback;
import app.pivo.android.basicsdkdemo.env.BorderedText;
import app.pivo.android.basicsdkdemo.env.ImageUtils;
import app.pivo.android.basicsdkdemo.env.Logger;
import app.pivo.android.basicsdkdemo.tflite.Classifier;
import app.pivo.android.basicsdkdemo.tflite.FeatureExtract;
import app.pivo.android.basicsdkdemo.tflite.YoloClassifier;
import app.pivo.android.basicsdkdemo.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener, View.OnClickListener {
    private static final Logger LOGGER = new Logger();

    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static boolean is_tiny = false;

    private static int TF_OD_API_INPUT_SIZE = 640;
    private static int TF_OD_API_INPUT_SIZE_ACC = 640;
    private static final String TF_OD_API_MODEL_FILE = "yolo-acc-6409_float16.tflite";
    private static int TF_OD_API_OUTPUT_SHAPE_ACC = 8400;

    private static final String TF_OD_API_MODEL_FILE_FAST = "yolo-lite-320_float16.tflite";
    private static int TF_OD_API_OUTPUT_SHAPE_FAST = 500;
    private static int TF_OD_API_INPUT_SIZE_FAST = 320;

    private static float CENTER_POSITION = TF_OD_API_INPUT_SIZE / 2;


    private static final String TF_OD_API_LABELS_FILE = "obj.names";

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API_FIRST = 0.75f;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.3f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(1920, 1080);
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private YoloClassifier detector;
    private YoloClassifier detector_acc;
    private YoloClassifier detector_fast;
    private FeatureExtract extractor;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;

    private BorderedText borderedText;

    private int FEATURE_INPUT_SIZE = 128;
    private int save_id_time = 10;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        try {
            detector_acc = new YoloClassifier(
                    getAssets(),
                    TF_OD_API_MODEL_FILE,
                    TF_OD_API_LABELS_FILE,
                    TF_OD_API_IS_QUANTIZED,
                    TF_OD_API_INPUT_SIZE_ACC,
                    TF_OD_API_OUTPUT_SHAPE_ACC,
                    8);
            detector_fast = new YoloClassifier(
                    getAssets(),
                    TF_OD_API_MODEL_FILE_FAST,
                    TF_OD_API_LABELS_FILE,
                    TF_OD_API_IS_QUANTIZED,
                    TF_OD_API_INPUT_SIZE_FAST,
                    TF_OD_API_OUTPUT_SHAPE_FAST,
                    8);

            detector = detector_acc;

            detector.setContext(this);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        extractor = new FeatureExtract(getAssets(),
                         "feature_map.tflite",
                             FEATURE_INPUT_SIZE,
                2048);

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
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

    Mat rotateMatrix = null;
    int prev_size = 0;
    Point center = null;
    Bitmap input_bitmap;
    Bitmap feature_bitmap;

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
        feature_bitmap = Bitmap.createBitmap(rgbFrameBitmap);


        Mat resize = new Mat();
        Utils.bitmapToMat(rgbFrameBitmap, resize);
        Imgproc.resize(resize, resize, new org.opencv.core.Size(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE));

        if (prev_size != TF_OD_API_INPUT_SIZE) {
            center = new Point(resize.height() / 2, resize.width() / 2);
            rotateMatrix = Imgproc.getRotationMatrix2D(center, sensorOrientation % 360 != 0 ? sensorOrientation - 180 : sensorOrientation - 0, 1.0);
            input_bitmap = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Config.ARGB_8888);
            prev_size = TF_OD_API_INPUT_SIZE;

            frameToCropTransform =
                    ImageUtils.getTransformationMatrix(
                            previewWidth, previewHeight,
                            TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
                            sensorOrientation, MAINTAIN_ASPECT);

            cropToFrameTransform = new Matrix();
            frameToCropTransform.invert(cropToFrameTransform);
        }

        Imgproc.warpAffine(resize, resize, rotateMatrix, new org.opencv.core.Size(resize.height(), resize.width()));
        Utils.matToBitmap(resize, input_bitmap);

        readyForNextImage();

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();

                        List<Classifier.Recognition> temp = new ArrayList<>();

                        try {
                            temp = detector.recognizeImage(input_bitmap);
                            temp = filter(feature_bitmap, temp);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                        List<Classifier.Recognition> results;

                        if (selectedObject.equals("all")) {
                            results = temp;
                        } else {
                            results = new ArrayList<>();

                            for (Classifier.Recognition record : temp) {
                                if (record.getTitle().equals(selectedObject)) results.add(record);
                            }
                        }
                        if (results.size() > 0) {
                            float position = results.get(0).getLocation().centerX();

                            runInBackground(() -> {
                                if (position < CENTER_POSITION - TF_OD_API_INPUT_SIZE / 10) {
                                    PivoSdk.getInstance().turnLeftContinuously((int) (30 * (1 - (CENTER_POSITION - position) / CENTER_POSITION)));
                                } else if (CENTER_POSITION + TF_OD_API_INPUT_SIZE / 10 < position) {
                                    PivoSdk.getInstance().turnRightContinuously((int) (30 * (1 - (position - CENTER_POSITION) / CENTER_POSITION)));
                                } else {
                                    PivoSdk.getInstance().stop();
                                }
                            });

                            runOnUiThread(() -> {
                                ((TextView) findViewById(R.id.object_center_position)).setText(position + "px");
                            });

                            Log.i("Position", "" + position);
                        } else {
                            PivoSdk.getInstance().stop();
                        }

                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        Log.e("CHECK", "run: " + results.size());

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();

                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null) {

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
                                        showCropInfo(TF_OD_API_INPUT_SIZE + "x" + TF_OD_API_INPUT_SIZE);
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });
                    }
                });
    }

    BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i("OpenCV", "OpenCV loaded successfully");

                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync("4.6.0", this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private int idSequence = 0;
    private Map<String, Object[]> featureMaps = new HashMap<>();
    private List<List<String>> saved_id_list = new ArrayList<>();

    public double getSimilarity(float[] featureMap1, float[] featureMap2) {
        if (featureMap1.length != featureMap2.length) {
            throw new IllegalArgumentException("Feature maps have different lengths");
        }
        double dotProduct = 0;
        double mag1 = 0;
        double mag2 = 0;
        for (int i = 0; i < featureMap1.length; i++) {
            dotProduct += featureMap1[i] * featureMap2[i];
            mag1 += featureMap1[i] * featureMap1[i];
            mag2 += featureMap2[i] * featureMap2[i];
        }
        double similarity = dotProduct / (Math.sqrt(mag1) * Math.sqrt(mag2));
        return similarity < 0 ? 0 : similarity;
    }

    private List<Classifier.Recognition> filter(Bitmap bitmap, List<Classifier.Recognition> detections) {
        Mat original = new Mat();
        bitmapToMat(bitmap, original);
        List<String> id_list = new ArrayList<>();

        for (Classifier.Recognition detection : detections) {
            float confidence = detection.getConfidence();
            RectF location = detection.getLocation();
            String title = detection.getTitle();

            if (confidence < MINIMUM_CONFIDENCE_TF_OD_API) continue;

            // 인식 된 부분의 ROI 영역 추출
            Rect roi_area = new Rect((int) (location.left / TF_OD_API_INPUT_SIZE * bitmap.getWidth()),
                                     (int) (location.top / TF_OD_API_INPUT_SIZE * bitmap.getHeight()),
                                     (int) (location.width() / TF_OD_API_INPUT_SIZE * bitmap.getWidth()),
                                     (int) (location.height() / TF_OD_API_INPUT_SIZE * bitmap.getHeight()));
            Mat image = original.submat(roi_area);
            Imgproc.resize(image, image, new org.opencv.core.Size(FEATURE_INPUT_SIZE, FEATURE_INPUT_SIZE), Imgproc.INTER_LINEAR);

            Bitmap bitmap_image = Bitmap.createBitmap(image.width(), image.height(), Config.ARGB_8888);
            matToBitmap(image, bitmap_image);

            float[] feature_map = extractor.getFeature(bitmap_image);

            double maxSimular = 0.50;
            String selectId = "";
            String simularity_temp = "";

            for (String key : featureMaps.keySet()) {
                float[] prev = (float[]) featureMaps.get(key)[0];
                String feature_title = (String) featureMaps.get(key)[1];
                long val_temp = 0;
                double val;

                if (!feature_title.equals(title)) continue;

                val = getSimilarity(prev, feature_map);
                val = val < 0 ? 0 : val;

                // 유사도 기반으로 기존 ID 검색
                if (val > maxSimular) {
                    maxSimular = val;
                    selectId = key;
                    simularity_temp = "" + val;
                }
            }

            boolean confidence_flag = false;

            // ID가 검색 되지 않은 경우 다음 번호로 ID 할당
            if ("".equals(selectId)) {
                selectId = "" + idSequence++;
                featureMaps.put(selectId, new Object[]{ feature_map, title });
            }
            else {
                // 일정 시간 이내에 id가 검출되지 않으면 초기화
                for (List<String> ids : saved_id_list) {
                    for (String id : ids) {
                        if (id.equals(selectId)) {
                            confidence_flag = true;
                            break;
                        }
                    }
                    if (confidence_flag) break;
                }
            }

            if (confidence_flag) {
                if (confidence < MINIMUM_CONFIDENCE_TF_OD_API) continue;
            } else {
                if (confidence < MINIMUM_CONFIDENCE_TF_OD_API_FIRST) continue;
            }

            id_list.add(selectId);

            detection.setId(selectId); // + "S: " + simularity_temp);
        }
        if (saved_id_list.size() > save_id_time) {
            saved_id_list.remove(0);
        }
        saved_id_list.add(id_list);

        return detections;
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

    private String model_name = "accuracy";

    @Override
    public void onClick(View v) {
        super.onClick(v);
        switch(v.getId()) {
            case R.id.model_performance:
                model_name = "accuracy";
                TF_OD_API_INPUT_SIZE = TF_OD_API_INPUT_SIZE_ACC;
                detector = detector_acc;
                break;

            case R.id.model_speed:
                model_name = "fast";
                TF_OD_API_INPUT_SIZE = TF_OD_API_INPUT_SIZE_FAST;
                detector = detector_fast;
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

        ((TextView)findViewById(R.id.selected_object)).setText(selectedObject);
        ((TextView)findViewById(R.id.select_model)).setText(model_name);
    }
}
