#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
use std::{io::Cursor, path::Path};

#[cfg(not(target_arch = "wasm32"))]
use kira::sound::streaming::{StreamingSoundHandle, StreamingSoundData, StreamingSoundSettings};
use kira::{Volume, sound::{EndPosition, FromFileError, PlaybackPosition, Region, SoundData, static_sound::{StaticSoundData, StaticSoundSettings}}};
use rand::Rng;

use crate::gputil::AssetSource;

pub fn load_static_sound(source: &impl AssetSource, path: impl AsRef<Path>, volume_db: f64) -> Result<StaticSoundData, FromFileError> {
    let bytes = source.get_bytes(path.as_ref()).map_err(FromFileError::IoError)?;
    let cursor = Cursor::new(bytes);
    let settings = StaticSoundSettings::default().volume(Volume::Decibels(volume_db));
    StaticSoundData::from_cursor(cursor, settings)
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub struct MusicData {
    data: Arc<[u8]>,
    volume_db: f64,
}

#[cfg(not(target_arch = "wasm32"))]
impl kira::sound::SoundData for MusicData {
    type Error = FromFileError;
    type Handle = StreamingSoundHandle<FromFileError>;

    fn into_sound(self) -> Result<(Box<dyn kira::sound::Sound>, Self::Handle), Self::Error> {
        use kira::sound::streaming::{StreamingSoundData, StreamingSoundSettings};

        let MusicData {data, volume_db} = self;
        let cursor = Cursor::new(data);
        let settings = StreamingSoundSettings::default().volume(Volume::Decibels(volume_db));
        let sound_data = StreamingSoundData::from_cursor(cursor, settings)?;
        sound_data.into_sound()
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_music_data(source: &impl AssetSource, path: impl AsRef<Path>, volume_db: f64) -> Result<MusicData, FromFileError> {
    let bytes = source.get_bytes(path.as_ref()).map_err(FromFileError::IoError)?;
    Ok(MusicData {data: Arc::from(bytes.into_owned()), volume_db})
}

#[cfg(target_arch = "wasm32")]
pub type MusicData = StaticSoundData;

#[cfg(target_arch = "wasm32")]
pub fn load_music_data(source: &impl AssetSource, path: impl AsRef<Path>, volume_db: f64) -> Result<MusicData, FromFileError> {
    let bytes = source.get_bytes(path.as_ref()).map_err(FromFileError::IoError)?;
    let cursor = Cursor::new(bytes);
    let settings = StaticSoundSettings::default().volume(Volume::Decibels(volume_db));
    StaticSoundData::from_cursor(cursor, settings)
}

pub type MusicHandle = <MusicData as SoundData>::Handle;



// Stores a set of effects packed into a single buffer
#[derive(Clone, Debug)]
pub struct SoundAtlas {
    pub sound: StaticSoundData,
    pub start_positions: Box<[f64]>,
}

impl SoundAtlas {
    pub fn num_sounds(&self) -> usize {
        self.start_positions.len()
    }

    pub fn get_sound(&self, n: usize) -> StaticSoundData {
        let start = PlaybackPosition::Seconds(self.start_positions[n]);
        let end = if n == self.start_positions.len() - 1 {
            EndPosition::EndOfAudio
        } else {
            EndPosition::Custom(PlaybackPosition::Seconds(self.start_positions[n + 1]))
        };

        self.sound.with_settings(self.sound.settings.playback_region(Region {start, end}))
    }

    pub fn random_sound(&self) -> StaticSoundData {
        let n = rand::thread_rng().gen_range(0..self.start_positions.len());
        self.get_sound(n)
    }

    pub fn load_with_starts(source: &impl AssetSource, path: impl AsRef<Path>, volume_db: f64, starts: &[f64]) -> Result<Self, FromFileError> {
        let sound = load_static_sound(source, path, volume_db)?;
        Ok(SoundAtlas {
            sound, start_positions: starts.to_vec().into_boxed_slice()
        })
    }

    pub fn load_with_stride(source: &impl AssetSource, path: impl AsRef<Path>, volume_db: f64, stride_s: f64) -> Result<Self, FromFileError> {
        let sound = load_static_sound(source, path, volume_db)?;
        let len_s = sound.duration().as_secs_f64();
        let mut starts = Vec::new();
        let mut t = 0.0;
        while t < len_s {
            t += stride_s;
            starts.push(t);
        }
        Ok(SoundAtlas {
            sound, start_positions: starts.into_boxed_slice()
        })
    }
}

