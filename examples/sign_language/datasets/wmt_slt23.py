import srt
import datetime
import pandas as pd
import re
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from functools import partial
from multiprocessing import Pool

from pose_format.pose import Pose

from . import SignLanguageDataset, register_dataset


def miliseconds_to_frame_index(miliseconds: int, fps: int) -> int:
    """ From https://github.com/bricksdont/sign-sockeye-baselines/ """
    return int(fps * (miliseconds / 1000))


def convert_srt_time_to_frame(srt_time: datetime.timedelta, fps: int) -> int:
    """ From https://github.com/bricksdont/sign-sockeye-baselines/ """
    seconds, microseconds = srt_time.seconds, srt_time.microseconds
    miliseconds = int((seconds * 1000) + (microseconds / 1000))
    return miliseconds_to_frame_index(miliseconds=miliseconds, fps=fps)


def process_video(subtitles_dir: Path, signs_dir: Path, video_id: str) -> pd.DataFrame:
    srt_filename = (subtitles_dir / f"{video_id}.srt")
    signs_filename = (signs_dir / f"{video_id}.pose")

    has_subtitles = srt_filename.is_file()
    has_signs = signs_filename.is_file()
    assert has_subtitles or has_signs

    def get_pose_length(pose_file: Path) -> int:
        with open(pose_file.as_posix(), "rb") as f:
            p = Pose.read(f.read())
        return p.body.data.shape[0]

    def get_pose_fps(pose_file: Path) -> int:
        with open(pose_file.as_posix(), "rb") as f:
            p = Pose.read(f.read())
        return p.body.fps

    df = pd.DataFrame(columns=SignLanguageDataset.MANIFEST_COLUMNS)
    if has_subtitles:
        fps = get_pose_fps(signs_filename) if has_signs else None
        with open(srt_filename, 'r') as srt_file:
            for subtitle in srt.parse(srt_file.read()):
                if subtitle.content.strip() == "":
                    print(f"Skipping empty subtitle: {subtitle}")
                    continue
                sample = {
                    "id": f"{video_id}-{subtitle.index}",
                    "translation": subtitle.content.replace('\t', ' '),
                }
                if has_signs:
                    start_frame = convert_srt_time_to_frame(subtitle.start, fps)
                    end_frame = convert_srt_time_to_frame(subtitle.end, fps)
                    if start_frame >= end_frame:
                        print(f"Skipping subtitle where start frame is equal or higher than end frame: {subtitle}")
                        continue
                    sample.update({
                        "signs_file": signs_filename.as_posix(),
                        "signs_offset": start_frame,
                        "signs_length": end_frame - start_frame,
                    })
                sample = pd.DataFrame.from_dict({k: [v] for k, v in sample.items()})
                df = pd.concat((df, sample), ignore_index=True)
    else:
        df = pd.DataFrame(columns=SignLanguageDataset.MANIFEST_COLUMNS)
        sample = {
            "id": video_id,
            "signs_file": signs_filename.as_posix(),
            "signs_offset": 0,
            "signs_length": get_pose_length(signs_filename),
        }
        sample = pd.DataFrame.from_dict({k: [v] for k, v in sample.items()})
        df = pd.concat((df, sample), ignore_index=True)

    return df


@register_dataset('wmt_slt23')
class WMT_SLT23(SignLanguageDataset):
    SPLITS = ['signsuisse', 'srf_monolingual', 'srf_parallel','test_dsgs-de']
    SIGNS_TYPES = ['mediapipe', None]
    SIGNS_LANGS = ['dsgs', None]
    TRANSLATION_LANGS = ['de', None]

    def __init__(self, base_dir: Path, split: str, signs_type: Optional[str] = None,
                 signs_lang: Optional[str] = None, translation_lang: Optional[str] = None) -> None:
        super().__init__(base_dir, split, signs_type, signs_lang, translation_lang)

        print(f"Loading {split} data from {base_dir}...")
        if split == 'signsuisse':
            csv_orginal = self.base_dir / "signsuisse" / \
                        "metadata_train.csv" 
            tsv_orginal = self.base_dir / \
                        "signsuisse_metadata.tsv" 
            if tsv_orginal not in self.base_dir.glob('*.tsv'):
                with open(csv_orginal, "r") as f, open(tsv_orginal, "w") as f_w:
                    f.readline()
                    f_w.write('id\tsigns_file\tsigns_offset\tsigns_length\tsigns_type\tsigns_lang\ttranslation\ttranslation_lang\tglosses\ttopic\tsigner_id\n')
                    lines = f.readlines()
                    for raw_line in lines:
                        line = re.split(r',\s*(?=(?:[^"]*"[^"]*")*[^"]*$)', raw_line.strip('\n'))
                        if (translation_lang in line[2]) and (signs_lang in line[3]):
                            with open(f'{self.base_dir}/signsuisse/example_mediapipe/{line[0]}.pose', 'rb') as f_pose:
                                p = Pose.read(f_pose.read())
                                f_w.write(f'{line[0]}\t{self.base_dir}/signsuisse/example_mediapipe/{line[0]}.pose\t0\t{p.body.data.shape[0]}\tmediapipe\tdsgs\t{line[7]}\tde\tNone\tNone\tNone\n')
                                # f_w.write(f'{line[0]}\t{self.base_dir}/lexicon_mediapipe/{line[0]}.pose\t0\t{line[10]}\tmediapipe\tdsgs\t{line[7]}\tde\tNone\tNone\tNone\n')
            self.data = pd.read_csv(tsv_orginal.as_posix(), sep='\t')
        elif ('signsuisse' in split) and ('srf' in split):
            split_subsplit = split.split('_')
            split_ = split_subsplit[0] if len(split_subsplit) > 1 else split
            subsplit = split_subsplit[1] if len(split_subsplit) > 1 else ""

            subtitles_dir = self.base_dir / split_ / subsplit / "subtitles"
            signs_dir = self.base_dir / split_ / subsplit / (signs_type if signs_type else "x")

            video_ids = [f.stem for f in subtitles_dir.glob('*.srt')] if subtitles_dir.is_dir() \
                            else [f.stem for f in signs_dir.glob('*.pose')]
            process_video_ = partial(process_video, subtitles_dir, signs_dir)
            with Pool() as pool:
                dfs = pool.map(process_video_, video_ids)        
            self.data = pd.concat(dfs, ignore_index=True)

            if not all(self.data['translation'].isna()):
                assert translation_lang is not None
                self.data['translation_lang'] = translation_lang
            if not all(self.data['signs_file'].isna()):
                assert signs_type is not None
                assert signs_lang is not None
                self.data['signs_type'] = signs_type
                self.data['signs_lang'] = signs_lang

            csv_orginal = self.base_dir / "signsuisse" / \
                        "metadata_train.csv" 
            tsv_orginal = self.base_dir / \
                        "signsuisse_metadata.tsv" 
            if tsv_orginal not in self.base_dir.glob('*.tsv'):
                with open(csv_orginal, "r") as f, open(tsv_orginal, "w") as f_w:
                    f.readline()
                    f_w.write('id\tsigns_file\tsigns_offset\tsigns_length\tsigns_type\tsigns_lang\ttranslation\ttranslation_lang\tglosses\ttopic\tsigner_id\n')
                    lines = f.readlines()
                    for raw_line in lines:
                        line = re.split(r',\s*(?=(?:[^"]*"[^"]*")*[^"]*$)', raw_line.strip('\n'))
                        if (translation_lang in line[2]) and (signs_lang in line[3]):
                            with open(f'{self.base_dir}/signsuisse/example_mediapipe/{line[0]}.pose', 'rb') as f_pose:
                                p = Pose.read(f_pose.read())
                                f_w.write(f'{line[0]}\t{self.base_dir}/signsuisse/example_mediapipe/{line[0]}.pose\t0\t{p.body.data.shape[0]}\tmediapipe\tdsgs\t{line[7]}\tde\tNone\tNone\tNone\n')
                                # f_w.write(f'{line[0]}\t{self.base_dir}/lexicon_mediapipe/{line[0]}.pose\t0\t{line[10]}\tmediapipe\tdsgs\t{line[7]}\tde\tNone\tNone\tNone\n')
            self.data = pd.concat([pd.read_csv(tsv_orginal.as_posix(), sep='\t'), self.data], ignore_index=True)
        else:
            split_subsplit = split.split('_')
            split_ = split_subsplit[0] if len(split_subsplit) > 1 else split
            subsplit = split_subsplit[1] if len(split_subsplit) > 1 else ""

            subtitles_dir = self.base_dir / split_ / subsplit / "subtitles"
            signs_dir = self.base_dir / split_ / subsplit / (signs_type if signs_type else "x")

            video_ids = [f.stem for f in subtitles_dir.glob('*.srt')] if subtitles_dir.is_dir() \
                            else [f.stem for f in signs_dir.glob('*.pose')]
            process_video_ = partial(process_video, subtitles_dir, signs_dir)
            with Pool() as pool:
                dfs = pool.map(process_video_, video_ids)        
            self.data = pd.concat(dfs, ignore_index=True)

            if not all(self.data['translation'].isna()):
                assert translation_lang is not None
                self.data['translation_lang'] = translation_lang
            if not all(self.data['signs_file'].isna()):
                assert signs_type is not None
                assert signs_lang is not None
                self.data['signs_type'] = signs_type
                self.data['signs_lang'] = signs_lang
