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
import java.util.Collections;
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
import app.pivo.android.basicsdkdemo.tracking.Function;
import app.pivo.android.basicsdkdemo.tracking.Measurement;
import app.pivo.android.basicsdkdemo.tracking.MultiBoxTracker;
import app.pivo.android.basicsdkdemo.tracking.Sort;
import app.pivo.android.basicsdkdemo.tracking.TrackedObject;


/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener, View.OnClickListener {
    private static final Logger LOGGER = new Logger();

    private static int TF_OD_API_INPUT_SIZE = 416;
    private static float CENTER_POSITION = TF_OD_API_INPUT_SIZE / 2;
    private static int[] TF_OD_API_OUTPUT_SHAPE = {2535, 2535};
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "yolov4-tiny-416.tflite";

    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/obj.names";

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;   //0.6f
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

    app.pivo.android.basicsdkdemo.tracking.Matrix sensorNoise = app.pivo.android.basicsdkdemo.tracking.Matrix.identity(2,2);
    app.pivo.android.basicsdkdemo.tracking.Matrix processNoise = app.pivo.android.basicsdkdemo.tracking.Matrix.identity(4,4);
    Function f = (Function) null;//by specifying null, you will use default function
    double minimumIOU = 0.65;
    double newShapeWeight = 0.1;
    double threshold = 8;
    double frame = 0;
    app.pivo.android.basicsdkdemo.tracking.Matrix initialVelocity = app.pivo.android.basicsdkdemo.tracking.Matrix.zero(2,1);
    app.pivo.android.basicsdkdemo.tracking.Matrix initialCovariance = app.pivo.android.basicsdkdemo.tracking.Matrix.identity(4,4);
    Sort s = new Sort(sensorNoise,processNoise,f,minimumIOU,newShapeWeight,threshold,initialVelocity,initialCovariance);

    class Detector {
        public Detector() {

        }
    }

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
    private List<Classifier.Recognition> previous = new ArrayList<>();
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
                        Log.i("croppedBitmap", "" + croppedBitmap);

                        List<Classifier.Recognition> temp = new ArrayList<>();
                        try {
                            temp = detector.recognizeImage(croppedBitmap);
                            //    previous = temp;
                        } catch (Exception e) {
                            //temp = previous;
                        }
                        List<Classifier.Recognition> results;
                        if (selectedObject.equals("all")) {
                            results = temp;
                        } else {
                            results = new ArrayList<>();

                            for (Classifier.Recognition record: temp) {
                                if (record.getTitle().equals(selectedObject)) results.add(record);
                            }
                        }





                       // Measurement[] tmp = new Measurement[arr_dum.size()];
                       // for(int i=0; i<arr_dum.size(); i++){
                       //     tmp[i] = arr_dum.get(i);//}

                        //s.predictPhase(currTimestamp);
                        //s.updatePhase(tmp,currTimestamp);
                        //TrackedObject[] track_result = s.getTracked();







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
                                paint.setTextSize(50);
                                canvas.drawText("test", 10, 100, paint);

                                cropToFrameTransform.mapRect(location);

                                result.setLocation(location);
                                mappedRecognitions.add(result);
                           }
                        }



                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        trackingOverlay.postInvalidate();

                        ArrayList<Measurement> arr_dum = new ArrayList<Measurement>();
                        for (final Classifier.Recognition potential : mappedRecognitions){
                            float x = potential.getLocation().left;
                            float y = potential.getLocation().top;
                            float width = potential.getLocation().width();
                            float height = potential.getLocation().height();
                            Measurement m = new Measurement(x,y,width,height,potential.getTitle(),0);
                            arr_dum.add(m);
                        }

                        Measurement[] tmp = new Measurement[arr_dum.size()];
                        for(int i=0; i<arr_dum.size(); i++){
                            tmp[i] = arr_dum.get(i);}
                        s.predictPhase(frame);
                        s.updatePhase(tmp,frame);
                        TrackedObject[] track_result = s.getTracked();
                        frame = frame + 1;
                        ArrayList<Integer> name = new ArrayList<Integer>();


                        for (final TrackedObject potential : track_result){
                            String name_raw = potential.name.substring(3);
                            int name_casted = Integer.parseInt(name_raw);
                            name.add(name_casted);

                        }



                        /*

                        SORT, 어떤 기준으로 object를 tracking할지 이하구문에서 설정합니다 !!!!!

                         */
                        if (name.size() > 0) {
                            int min = Collections.min(name);  // 어떤 object를 따라갈것인지 지정합니다. min이면 먼저들어온 object, max면 가장 마지막에 들어온 object
                            int pivot_object = name.indexOf(min);
                            float position = (float) track_result[pivot_object].position.get(0,0) + (float) track_result[pivot_object].width/2;
                            runInBackground(() -> {
                                if (position < CENTER_POSITION - 20) {
                                    PivoSdk.getInstance().turnLeftContinuously((int)(30 * (1 - (CENTER_POSITION - position) / CENTER_POSITION)));
                                } else if (CENTER_POSITION + 20 < position) {
                                    PivoSdk.getInstance().turnRightContinuously((int)(30 * (1 - (position - CENTER_POSITION) / CENTER_POSITION)));
                                } else {

                                    PivoSdk.getInstance().stop();
                                }
                            });

                            runOnUiThread(() -> {
                                ((TextView)findViewById(R.id.object_center_position)).setText(position + "px");
                            });

                            Log.i("Position", "" + position);
                        } else {
                            PivoSdk.getInstance().stop();
                        }




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
                output_shape = new int[]{ 2535, 2535 };
                input_size = 416;
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
        Bitmap tempCropped = croppedBitmap;
        Matrix tempFrame = frameToCropTransform;
        Matrix tempCrop = cropToFrameTransform;
        try {
            TF_OD_API_INPUT_SIZE = input_size;
            croppedBitmap = Bitmap.createBitmap(input_size, input_size, Config.ARGB_8888);
            Log.i("nCroppedBitmap", "" + croppedBitmap);

            frameToCropTransform =
                    ImageUtils.getTransformationMatrix(
                            previewWidth, previewHeight,
                            input_size, input_size,
                            sensorOrientation, MAINTAIN_ASPECT);

            cropToFrameTransform = new Matrix();
            frameToCropTransform.invert(cropToFrameTransform);

            detector = YoloV4Classifier.create(
                    getAssets(),
                    modelSelect,
                    TF_OD_API_LABELS_FILE,
                    TF_OD_API_IS_QUANTIZED,
                    input_size,
                    output_shape,
                    true);

        } catch (IOException e) {
            e.printStackTrace();
            detector = temp;
            croppedBitmap = tempCropped;
            frameToCropTransform = tempFrame;
            cropToFrameTransform = tempCrop;
        }

        ((TextView)findViewById(R.id.selected_object)).setText(selectedObject);
        ((TextView)findViewById(R.id.select_model)).setText(model_name);

    }
}


