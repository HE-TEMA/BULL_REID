# Copyright © 2026 Bull® SAS. All rights reserved

"""Person re-identification service using image detections and NGSI-LD.

This module provides a FastAPI application that processes object detections,
extracts person appearance embeddings, associates detections across images,
stores image-related data in MinIO, and publishes enriched detection entities
to an NGSI-LD context broker.

The service supports both legacy image notifications and notifications
containing precomputed object detections.
"""

import cv2
import numpy as np
from pathlib import Path
import torch

from boxmot import DeepOcSort
from ultralytics import YOLO

from boxmot.appearance.reid_auto_backend import ReidAutoBackend

from minio import Minio
from minio.error import S3Error
import requests
import json
from fastapi import FastAPI, Request
import random
from pydantic import BaseModel, HttpUrl, ValidationError, field_validator
from typing import List, Optional, Union
import os
import datetime
import time



app = FastAPI()
BASE_PATH = os.environ.get("BASE_PATH", "/")
DEBUG = os.environ.get("DEBUG", "/")

if DEBUG:
    print("Verbose mode enabled: Debugging information will be shown.")

main_entity = "urn:ngsi-ld:tema:ATOS:TFA-07:PersonREID:"
counter = 1
random_session = random.randint(1, 999)


euclidean_distance = 0.22


# Function to increment the counter
def increment_counter():
    """Increment the global entity identifier counter by one."""
    global counter
    counter += 1


# Function to get the current value of the counter
def get_counter():
    """Return the current global entity identifier counter.

    Returns:
        int: The current counter value.
    """
    global counter
    return counter


def get_session():
    """Generate a timestamp string representing the current local time.

    Returns:
        str: The current timestamp formatted as YYYYMMDDHHMMSS.
    """
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def create_unique_id():
    """Generate an entity identifier using the configured namespace and time.

    The identifier combines the global entity namespace with the current
    timestamp. The counter is not incremented by this function.

    Returns:
        str: A timestamp-based entity identifier.
    """
    session_id = get_session()
    return f"{main_entity}_{session_id}"


class MinioClient:
    """Provide file and bucket operations for a MinIO object storage service.

    Args:
        endpoint (str): MinIO server hostname and port.
        access_key (str): Access key used to authenticate with MinIO.
        secret_key (str): Secret key used to authenticate with MinIO.
        secure (bool): Whether to use a secure HTTPS connection.
    """

    def __init__(self, endpoint, access_key, secret_key, secure=True):
        """Initialize the MinIO client with the supplied credentials."""
        self.client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure
        )

    def create_bucket(self, bucket_name):
        """Create a bucket if it does not already exist.

        Args:
            bucket_name (str): Name of the bucket to create.

        Raises:
            No exception is propagated for MinIO S3 errors; they are logged.
        """
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                print(f"Bucket '{bucket_name}' created successfully.")
            else:
                print(f"Bucket '{bucket_name}' already exists.")
        except S3Error as e:
            print(f"Error creating bucket: {e}")

    def upload_file(self, bucket_name, object_name, file_path):
        """Upload a local file to a MinIO bucket.

        Args:
            bucket_name (str): Destination bucket name.
            object_name (str): Object key under which the file is stored.
            file_path (str): Path to the local file to upload.

        Returns:
            bool: True if the upload succeeds, otherwise False.
        """
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            print(
                f"File '{file_path}' uploaded to '{bucket_name}/{object_name}' successfully."
            )
            return True
        except S3Error as e:
            print(f"Error uploading file: {e}")
            return False

    def download_file(self, bucket_name, object_name, file_path):
        """Download an object from MinIO to a local file.

        Args:
            bucket_name (str): Source bucket name.
            object_name (str): Object key to download.
            file_path (str): Local destination path.

        Returns:
            bool: True if the download succeeds, otherwise False.
        """
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            print(
                f"File '{object_name}' downloaded from '{bucket_name}' to '{file_path}' successfully."
            )
            return True
        except S3Error as e:
            print(f"Error downloading file: {e}")
            return False

    def list_objects(self, bucket_name):
        """Print the names of objects stored in a bucket.

        Args:
            bucket_name (str): Name of the bucket to inspect.

        Raises:
            No exception is propagated for MinIO S3 errors; they are logged.
        """
        try:
            objects = self.client.list_objects(bucket_name)
            print(f"Objects in bucket '{bucket_name}':")
            for obj in objects:
                print(obj.object_name)
        except S3Error as e:
            print(f"Error listing objects: {e}")


class REID:
    """Track detected people and associate them across images using embeddings.

    The class combines DeepOCSort tracking with a re-identification backend.
    Appearance embeddings are compared using Euclidean distance to associate
    detections with previously observed identities.

    Attributes:
        euclidean_distance (float): Maximum embedding distance accepted
            for a potential identity match.
        reid_model_path (str): Path to the re-identification model weights.
        detection_model_path (str): Path to the object detection model.
        device (torch.device): Device used for inference.
        next_id (int): Next locally assigned detection identifier.
        tracker: DeepOCSort tracker instance.
        reid_backend: Re-identification backend wrapper.
        reid_model: Initialized model used to extract embeddings.
        track_detections (list or numpy.ndarray): Stored detections and IDs.
        dets_embs (list or numpy.ndarray): Embeddings corresponding to
            the stored detections.
    """

    def __init__(self, reid_model_path, detection_model_path, distance=0.22):
        """Initialize the tracker and person re-identification model.

        Args:
            reid_model_path (str): Path to the re-identification weights.
            detection_model_path (str): Path to the detection model weights.
            distance (float): Maximum Euclidean distance used to match
                appearance embeddings.
        """
        self.euclidean_distance = distance
        self.reid_model_path = reid_model_path
        self.detection_model_path = detection_model_path

        self.device = torch.device("cuda", 0)
        self.next_id = 1

        # Tracker Model
        self.tracker = DeepOcSort(
            reid_weights=Path(self.reid_model_path),  # which ReID model to use
            device=self.device,
            half=False,
        )

        # REID Model
        self.reid_backend = ReidAutoBackend(
            weights=Path(self.reid_model_path), device=self.device, half=False
        )
        self.reid_model = self.reid_backend.get_backend()

        self.track_detections = []
        self.dets_embs = []

    def get_next_id(self):
        """Return the next locally assigned detection identifier.

        Returns:
            int: The next identifier to assign.
        """
        return self.next_id
    
    def get_len_dets_emb(self):
        """Return the number of stored appearance embeddings.

        Returns:
            int: Number of embeddings currently stored.
        """
        return len(self.dets_embs)
    
    def set_next_id(self, value):
        """Set the next locally assigned detection identifier.

        Args:
            value (int): Identifier to assign as the next ID.
        """
        self.next_id = value

    def calculate_emb_distance(self, emb1, emb2):
        """Calculate the Euclidean distance between two embeddings.

        Args:
            emb1 (numpy.ndarray): First appearance embedding.
            emb2 (numpy.ndarray): Second appearance embedding.

        Returns:
            numpy.float64 or float: Euclidean distance between the embeddings.
        """
        emb_distance = 0
        emb_distance = np.linalg.norm(emb1 - emb2)
        return emb_distance

    def tracks_to_json(self, tracks):
        """Convert tracking results into JSON-serializable detection objects.

        Each track is expected to contain bounding-box coordinates and a
        tracking identifier. Coordinates are converted into x, y, width,
        and height format.

        Args:
            tracks (iterable): Tracks containing at least
                [x1, y1, x2, y2, track_id].

        Returns:
            list[dict]: Detection dictionaries containing an ID and bounding
            box, ready for JSON serialization.
        """
        # Create an empty list to store the detections
        detections = []

        # Iterate over each track
        for track in tracks:
            # Extract bbox coordinates and calculate width and height
            bbox = {
                "x": track[0],  # Top left x-coordinate
                "y": track[1],  # Top left y-coordinate
                "width": track[2] - track[0],  # width = bottom_right_x - top_left_x
                "height": track[3] - track[1],  # height = bottom_right_y - top_left_y
            }

            # Create the detection dictionary
            detection = {
                "id": str(track[4]),  # Convert id to string
                "bbox": bbox,  # Bounding box
                # "confidence": track[5],  # Confidence score
            }

            # Append the detection to the list
            detections.append(detection)

        return detections

    def merge_track_detections(self, new_track_detections, new_dets_embs):
        """Append tracks with previously unseen IDs to the stored detections.

        Existing track IDs are retained, and embeddings are appended for
        newly encountered IDs. The method updates both stored detections
        and their corresponding embeddings.

        Args:
            new_track_detections (numpy.ndarray): New detections containing
                bounding boxes, IDs, and confidence values.
            new_dets_embs (numpy.ndarray): Embeddings corresponding to the
                new detections.

        Returns:
            None
        """
        existing_ids = {track[4] for track in self.track_detections}

        # Prepare a list for the merged tracks
        merged_tracks = self.track_detections.copy()  # Start with the first group
        merged_embs = self.dets_embs.copy()

        # Iterate through the second group and add tracks with unique IDs
        for i, track in enumerate(new_track_detections):
            if track[4] not in existing_ids:
                merged_tracks = np.vstack((merged_tracks, track))
                merged_embs = np.vstack(
                    (merged_embs, new_dets_embs[i])
                )  # Append the corresponding embedding

        self.track_detections = merged_tracks
        self.dets_embs = merged_embs

    def UpdateReid(self, new_dets_embs, new_track_detections):
        """Associate new detections with previously stored identities.

        For each stored embedding, find the closest new embedding whose
        Euclidean distance is below the configured threshold. When a match
        is found, assign the previous identity to the corresponding new
        detection. Then merge previously unseen track IDs into the stored
        detections.

        Args:
            new_dets_embs (numpy.ndarray): Appearance embeddings extracted
                from the current image.
            new_track_detections (numpy.ndarray): Current detections,
                including bounding boxes and provisional IDs.

        Returns:
            numpy.ndarray: Current detections with matched IDs where
            applicable.

        Note:
            Matching is performed independently for each stored embedding;
            this method does not enforce one-to-one assignment globally.
        """
        if len(self.track_detections) == 0:

            if DEBUG:
                print("new embeddings")
                print(len(new_dets_embs))
            self.track_detections = new_track_detections
            self.dets_embs = new_dets_embs

            return new_track_detections

        else:
            if DEBUG:
                print("previous detection embeddings")
                print(len(self.dets_embs))
                print("new detection embeddings")
                print(len(new_dets_embs))
            for i in range(len(self.dets_embs)):
                found_match = False
                min_dist = 1
                index = 999
                for j in range(len(new_dets_embs)):
                    tmp_dist = self.calculate_emb_distance(
                        new_dets_embs[j], self.dets_embs[i]
                    )
                    if tmp_dist < self.euclidean_distance and tmp_dist < min_dist:
                        # update track on id_data
                        min_dist = tmp_dist
                        found_match = True
                        index = j

                if found_match:
                    if DEBUG:
                        print("found match")
                        print("previous id")
                        print(self.track_detections[i][4])
                        print("new id")
                        print(new_track_detections[index][4])

                    new_track_detections[index][4] = self.track_detections[i][4]

            self.merge_track_detections(new_track_detections, new_dets_embs)

            return new_track_detections
        
    def ProcessImage_boxes(self, input_image, boxes):
        """Process an image using externally supplied detection boxes.

        Filters detections to the person category, updates the tracker,
        extracts appearance embeddings, and associates detections with
        previously observed identities.

        Args:
            input_image (numpy.ndarray): Image in the format expected by
                the tracker and re-identification model.
            boxes (iterable): Detection objects containing bbox, category_id,
                and confidence attributes.

        Returns:
            numpy.ndarray or None: Processed tracks with assigned IDs,
            or None if no person detections are available.
        """
        detections = []
        for box in boxes:
            if(box.category_id == 0):
                x1, y1, x2, y2 = map(int, box.bbox)
                detections.append([x1,y1,x2,y2,box.confidence,box.category_id])

        if len(detections) > 0:
            detections = np.array(detections)

            # Update Tracker
            new_track_detections = self.tracker.update(detections, input_image)

            curr_id = self.get_next_id()
            next_id = curr_id + len(detections)
            self.set_next_id(next_id)
            ids = np.arange(curr_id, len(detections) + curr_id).reshape(-1, 1)
            new_detections = np.hstack((detections[:, :4], ids, detections[:, 4:5]))

            if DEBUG:
                print("raw detections")
                print(detections)
                print("new track detections ERR1")
                print(new_track_detections)
                print("new detections")
                print(new_detections)

            new_dets_embs = self.reid_model.get_features(
                new_detections[:, 0:4], input_image
            )

            output_tracks = self.UpdateReid(new_dets_embs, new_detections)
            return output_tracks
            

        else:
            return None

    def ProcessImage(self, input_image):
        """Detect and track people directly from an image.

        Runs the configured object detector, filters for the person class,
        updates the tracker, extracts appearance embeddings, and associates
        detections with previously observed identities.

        Args:
            input_image (numpy.ndarray): Image to process.

        Returns:
            numpy.ndarray or None: Processed tracks with assigned IDs,
            or None if no detections are available.
        """
        # Person Detection
        det_results = self.detection_model(input_image, classes=[0])
        det_boxes = det_results[0].boxes.xyxy.tolist()
        det_classes = det_results[0].boxes.cls.tolist()
        det_confidences = det_results[0].boxes.conf.tolist()

        if len(det_boxes) > 0:
            detections = []
            for i in range(len(det_boxes)):
                detections.append(
                    [
                        det_boxes[i][0],
                        det_boxes[i][1],
                        det_boxes[i][2],
                        det_boxes[i][3],
                        det_confidences[i],
                        det_classes[i],
                    ]
                )

            detections = np.array(detections)

            # Update Tracker
            new_track_detections = self.tracker.update(detections, input_image)


            curr_id = self.get_next_id()
            next_id = curr_id + len(detections)
            self.set_next_id(next_id)
            ids = np.arange(curr_id, len(detections) + curr_id).reshape(-1, 1)
            new_detections = np.hstack((detections[:, :4], ids, detections[:, 4:5]))


            if DEBUG:
                print("raw detections")
                print(detections)
                print("new track detections ERR1")
                print(new_track_detections)
                print("new detections")
                print(new_detections)

            new_dets_embs = self.reid_model.get_features(
                new_detections[:, 0:4], input_image
            )

            output_tracks = self.UpdateReid(new_dets_embs, new_detections)
            return output_tracks
        else:
            return None

    def GetTracks(self):
        """Return all detections currently stored by the re-identification system.

        Returns:
            numpy.ndarray or list: Stored tracking detections and their IDs.
        """
        return self.track_detections

# Message classes

class New_Box(BaseModel):
    bbox: List[float]
    category_id: int
    confidence: float
    object_id: int
    unique_id: int

class Box(BaseModel):
    bbox: List[float]
    category_id: int
    confidence: float
    object_id: int


class ClassCategory(BaseModel):
    category_id: int
    name: str

class Detection(BaseModel):
    boxes: List[Box]
    class_categories: List[ClassCategory]

class Parameters(BaseModel):
    About: Optional[str] = None
    AbsoluteAltitude: Optional[str] = None
    AlreadyApplied: Optional[bool] = None
    AltitudeType: Optional[str] = None
    Aperture: Optional[float] = None
    ApertureValue: Optional[float] = None
    BM_id: Optional[str] = None
    BitsPerSample: Optional[int] = None
    CamReverse: Optional[int] = None
    CameraSerialNumber: Optional[str] = None
    CircleOfConfusion: Optional[str] = None
    ColorComponents: Optional[int] = None
    ColorSpace: Optional[str] = None
    ComponentsConfiguration: Optional[str] = None
    Compression: Optional[str] = None
    Contrast: Optional[str] = None
    CreateDate: Optional[str] = None
    DateTimeOriginal: Optional[str] = None
    DigitalZoomRatio: Optional[Union[int, float]] = None
    Directory: Optional[str] = None
    DroneModel: Optional[str] = None
    DroneSerialNumber: Optional[str] = None
    EncodingProcess: Optional[str] = None
    ExifByteOrder: Optional[str] = None
    ExifImageHeight: Optional[int] = None
    ExifImageWidth: Optional[int] = None
    ExifToolVersion: Optional[float] = None
    ExifVersion: Optional[str] = None
    ExposureCompensation: Optional[Union[int, float]] = None
    ExposureMode: Optional[str] = None
    ExposureProgram: Optional[str] = None
    ExposureTime: Optional[str] = None
    FNumber: Optional[float] = None
    FOV: Optional[str] = None
    FileAccessDate: Optional[str] = None
    FileInodeChangeDate: Optional[str] = None
    FileModifyDate: Optional[str] = None
    FileName: Optional[str] = None
    FilePermissions: Optional[str] = None
    FileSize: Optional[str] = None
    FileSource: Optional[str] = None
    FileType: Optional[str] = None
    FileTypeExtension: Optional[str] = None
    Flash: Optional[str] = None
    FlashpixVersion: Optional[str] = None
    FlightPitchDegree: Optional[float] = None
    FlightRollDegree: Optional[Union[float,str]] = None
    FlightXSpeed: Optional[float] = None
    FlightYSpeed: Optional[float] = None
    FlightYawDegree: Optional[float] = None
    FlightZSpeed: Optional[float] = None
    FocalLength: Optional[str] = None
    FocalLength35efl: Optional[str] = None
    FocalLengthIn35mmFormat: Optional[str] = None
    Format: Optional[str] = None
    GPSAltitude: Optional[str] = None
    GPSAltitudeRef: Optional[str] = None
    GPSLatitude: Optional[str] = None
    GPSLatitudeRef: Optional[str] = None
    GPSLongitude: Optional[str] = None
    GPSLongitudeRef: Optional[str] = None
    GPSMapDatum: Optional[str] = None
    GPSPosition: Optional[str] = None
    GPSStatus: Optional[str] = None
    GPSVersionID: Optional[str] = None
    GainControl: Optional[str] = None
    GimbalPitchDegree: Optional[float] = None
    GimbalReverse: Optional[int] = None
    GimbalRollDegree: Optional[str] = None
    GimbalYawDegree: Optional[Union[float, str]] = None
    GpsStatus: Optional[str] = None
    HasCrop: Optional[bool] = None
    HasSettings: Optional[bool] = None
    HyperfocalDistance: Optional[str] = None
    ISO: Optional[int] = None
    ImageDescription: Optional[str] = None
    ImageHeight: Optional[int] = None
    ImageSize: Optional[str] = None
    ImageWidth: Optional[int] = None
    InteropIndex: Optional[str] = None
    InteropVersion: Optional[str] = None
    JFIFVersion: Optional[float] = None
    LensInfo: Optional[str] = None
    LensSerialNumber: Optional[str] = None
    LightSource: Optional[str] = None
    LightValue: Optional[float] = None
    MIMEType: Optional[str] = None
    Make: Optional[str] = None
    MaxApertureValue: Optional[float] = None
    Megapixels: Optional[float] = None
    MeteringMode: Optional[str] = None
    Modality: Optional[str] = None
    Model: Optional[str] = None
    ModifyDate: Optional[str] = None
    OffsetSchema: Optional[int] = None
    Orientation: Optional[str] = None
    RelativeAltitude: Optional[str] = None
    ResolutionUnit: Optional[str] = None
    RtkDiffAge: Optional[float] = None
    RtkFlag: Optional[int] = None
    RtkStdHgt: Optional[float] = None
    RtkStdLat: Optional[float] = None
    RtkStdLon: Optional[float] = None
    Saturation: Optional[str] = None
    ScaleFactor35efl: Optional[float] = None
    SceneCaptureType: Optional[str] = None
    SceneType: Optional[str] = None
    SelfData: Optional[str] = None
    SensitivityType: Optional[str] = None
    SerialNumber: Optional[str] = None
    Sharpness: Optional[str] = None
    ShutterCount: Optional[int] = None
    ShutterSpeed: Optional[str] = None
    ShutterSpeedValue: Optional[str] = None
    ShutterType: Optional[str] = None
    Software: Optional[str] = None
    SourceFile: Optional[str] = None
    SurveyingMode: Optional[int] = None
    ThumbnailLength: Optional[int] = None
    ThumbnailOffset: Optional[int] = None
    UTCAtExposure: Optional[str] = None
    Version: Optional[int] = None
    Warning: Optional[str] = None
    WhiteBalance: Optional[str] = None
    XPComment: Optional[str] = None
    XPKeywords: Optional[str] = None
    XResolution: Optional[int] = None
    YCbCrPositioning: Optional[str] = None
    YCbCrSubSampling: Optional[str] = None
    YResolution: Optional[int] = None
    flight_number: Optional[str] = None
    uav_id: Optional[str] = None
    origin_minio_url: Optional[str] = None

    model_config = {"extra": "allow"}  # allow dynamic DJI fields


class PersonVehicleDetection(BaseModel):
    BM_id: str
    bucket: str
    creator: str
    description: str
    detection: Detection
    flight_number: str
    id: str
    parameters: Parameters
    title: str
    type: str
    uav_id: str
    origin_minio_url: str
    @field_validator('*', mode='before')
    def unwrap_property(cls, v):
        if isinstance(v, dict) and 'value' in v:
            return v['value']
        return v

class Notification(BaseModel):
    id: str
    type: str
    subscriptionId: str
    notifiedAt: str
    data: List[PersonVehicleDetection]


def createOrionEntity(orion_ld_endpoint, entity):
    """Submit an NGSI-LD entity to the configured context broker.

    Sends an HTTP POST request to the broker's entity creation endpoint,
    including the JSON-LD content negotiation headers.

    Args:
        orion_ld_endpoint (str): Base URL of the NGSI-LD context broker.
        entity (dict): Entity payload to serialize and submit.

    Returns:
        bool: True if the broker responds with a successful HTTP status;
            False if the request fails or the broker returns an error.
    """
    h = {
        "Content-Type": "application/json",
        "Accept": "application/ld+json",
        # "Fiware-Service": "hackathon",
        "Link": '<https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
    }
    # {"content-type": "application/json"}
    try:
        response = requests.post(
            url=f"{orion_ld_endpoint}/ngsi-ld/v1/entities",
            headers=h,
            data=json.dumps(entity),
        )
        response.raise_for_status()  # Raise an error for bad status codes
        print(f"Entity created successfully with status code: {response.status_code}")
        return True
    except requests.RequestException as e:
        print(f"Failed to create entity: {e}")
        return False


@app.post(f"{BASE_PATH}/test")
# @app.post("/test")
async def test():
    """Provide a basic endpoint for testing service availability."""
    print("test")


@app.post(f"{BASE_PATH}/image")
async def image_test():
    """Run the re-identification pipeline on predefined local test images.

    Processes two sample images, serializes their detections, and writes
    per-image and accumulated tracking results to JSON files.

    Returns:
        None
    """
    image = cv2.imread("resized_DJI_20211023131938_0059_Z.JPG")
    tracks1 = reid.ProcessImage(image)
    dets1 = reid.tracks_to_json(tracks1)
    if DEBUG:
        print("tracks1")
        print(dets1)

    file_path = "test1.json"

    data = {"output": {"detections": dets1}}

    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    image = cv2.imread("resized_DJI_20211023131941_0060_Z.JPG")
    tracks2 = reid.ProcessImage(image)
    dets2 = reid.tracks_to_json(tracks2)

    if DEBUG:
        print("tracks2")
        print(dets2)

    file_path = "test2.json"

    data = {"output": {"detections": dets2}}

    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    total_tracks = reid.GetTracks()
    total_dets = reid.tracks_to_json(total_tracks)

    file_path = "test.json"

    data = {"output": {"detections": total_dets}}

    if DEBUG:
        print ("output")

    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


@app.post(f"{BASE_PATH}/notification_test")
async def json_input(request: Request):
    """Validate an incoming notification payload without processing it.

    Reads the request body as JSON, validates it against the Notification
    model, and prints the validated notification.

    Args:
        request (Request): Incoming FastAPI request containing JSON data.

    Returns:
        dict: A status response when validation succeeds.

    Raises:
        ValidationError: If the request payload does not match the schema.
    """
    raw = await request.json()
    try:
        subs = Notification(**raw)
    except ValidationError as e:
        print(e.json())  # Full validation error with details
        raise
    print (subs)
    return {"status": "ok"}

@app.post(f"{BASE_PATH}/notification")
async def person_reid(subs: Notification):
    """Process a notification containing precomputed object detections.

    Downloads the source image from MinIO, extracts person appearance
    embeddings, associates detections with re-identification IDs, and
    publishes an enriched NGSI-LD entity to the context broker. Processing
    timings and detection statistics are appended to a local log file.

    Args:
        subs (Notification): Validated notification containing detection
            boxes, image metadata, and flight information.

    Returns:
        None

    Note:
        This endpoint uses the first item in the notification's data list.
        Image retrieval or entity creation failures are logged by the
        relevant helper functions.
    """
    start_time = datetime.datetime.now()
    number_of_people_entity = 0
    number_of_people_database = 0

    uav_id = subs.data[0].uav_id
    flight_id = subs.data[0].flight_number
    image_name = subs.data[0].parameters.FileName
    bm = subs.data[0].BM_id

    origin_minio_url = subs.data[0].origin_minio_url

    if "/auth/" in origin_minio_url:
        object_name = origin_minio_url.split("/auth/")[1]
        bucket="auth"
    elif "/use/" in origin_minio_url:
        object_name = origin_minio_url.split("/use/")[1]
        bucket = "use"
    else:
        object_name = None  # or raise an error / handle as you wish

    boxes = subs.data[0].detection.boxes

    time_before_downloading_image_sec = datetime.datetime.now()
    time_before_downloading_image = time.time_ns()
    time_after_downloading_image = time_before_downloading_image

    ret = minio_client.download_file(bucket, object_name, "dummy2.jpg")
    if ret:
        print("downloaded image correctly")
        input_image = cv2.imread("dummy2.jpg")

        time_after_downloading_image_sec = datetime.datetime.now()
        time_after_downloading_image = time.time_ns()
        time_downloading_image_sec = (time_after_downloading_image_sec - time_before_downloading_image_sec).total_seconds()
        time_downloading_image = time_after_downloading_image - time_before_downloading_image

        if DEBUG:
            print("input detections")
            print(boxes)

        new_boxes = []

        for box in boxes:
            new_box = New_Box(bbox=box.bbox, category_id=box.category_id, object_id=box.object_id, confidence=box.confidence, unique_id=-1)
            new_boxes.append(new_box)

            if box.category_id == 0: # <- Only for data logs
                number_of_people_entity +=1

        number_of_people_database = reid.get_len_dets_emb()

        time_before_reid_sec = datetime.datetime.now()
        time_before_reid = time.time_ns()

        tracks = reid.ProcessImage_boxes(input_image,boxes)

        time_after_reid_sec = datetime.datetime.now()
        time_after_reid = time.time_ns()

        time_reid_sec = (time_after_reid_sec - time_before_reid_sec).total_seconds()
        time_reid = time_after_reid - time_before_reid

        if input_image is not None:
            height, width = input_image.shape[:2]
            image_size = {"width": width, "height": height}
        else:
            image_size = None


        if tracks is None:
            print("No person detections")
        else:

            #Update new boxes with the unique_id
            for new_box in new_boxes:
                x1, y1, x2, y2 = map(int, new_box.bbox)
                for track in tracks:
                    x1_t, y1_t, x2_t, y2_t, unique_id_t, confidence_t = track
                    if(x1 == x1_t and y1 == y1_t and x2 == x2_t and y2 == y2_t):
                        new_box.unique_id = int(unique_id_t)
                        break


        current_id = create_unique_id()


        new_boxes_dict = [box.model_dump() for box in new_boxes]
        class_categories_dict = [cat.model_dump() for cat in subs.data[0].detection.class_categories]
        parameters_dict = [subs.data[0].parameters.model_dump()]

        entity = {
            "id": current_id,
            "type": os.environ.get("BROKER_TYPE_ID"),
            "bm_id": {
                "type": "Property",
                "value": subs.data[0].BM_id,
            },
            "bucket": {
                "type": "Property",
                "value": bucket_name,
            },
            "creator": {
                "type": "Property",
                "value": "ATOS",
            },
            "description": {
                "type": "Property",
                "value": "Detections of the objects in the image, with unique id",
            },
            "detection": {
                "type": "Property",
                "value": {
                    "boxes": new_boxes_dict,
                    "class_categories": class_categories_dict,
                },
            },
            "flight_number": {
                "type": "Property",
                "value": subs.data[0].flight_number,
            },
            "origin_minio_url": {
                "type": "Property",
                "value": subs.data[0].origin_minio_url,
            },
            "parameters" :{
                "type": "Property",
                "value": parameters_dict,
            },
            "title":{
                "type": "Property",
                "value": "Object Detection + REID",
            },
            "uav_id":{
                "type": "Property",
                "value": subs.data[0].uav_id,
            }
        }

        entity_created = createOrionEntity(orion_ld_endpoint, entity)
        if entity_created:
            print("NGSI-LD entity creation succeeded")
        else:
            print("NGSI-LD entity creation failed")

        end_time = datetime.datetime.now()

        time_total = (end_time - start_time).total_seconds()

        timelapse_log = {
            "total_seconds": time_total,
            "dowloading_image_time (ns)": time_downloading_image,
            "time_reid (ns)": time_reid,
            "downloading_image_time (seconds)": time_downloading_image_sec,
            "time_reid (seconds)": time_reid_sec,
            "number_of_people_in_database": number_of_people_database,
            "number_of_people_detected": number_of_people_entity,
            "image_size": image_size
        }

        with open("data_tracking.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps(timelapse_log) + "\n")

    else:
        print("download failed")


if __name__ == "__main__":
    bucket_name = "atos"
    object_name = "example-object"
    endpoint = os.environ.get("endpoint")
    access_key = os.environ.get("access_key")
    secret_key = os.environ.get("secret_key")

    orion_ld_endpoint = (
        os.environ.get("BROKER_URL")
        # "https://orion.tema.digital-enabler.eng.it"  # Orion-LD server endpoint
    )

    minio_client = MinioClient(endpoint, access_key, secret_key, secure=True)

    reid = REID("osnet_x0_25_market1501.pt", "yolov8n.pt")

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT")))
