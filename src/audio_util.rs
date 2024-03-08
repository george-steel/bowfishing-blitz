use kira::{sound::{static_sound::{StaticSoundData, StaticSoundSettings}, EndPosition, FromFileError, PlaybackPosition, Region}, Volume};
use rand::Rng;

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

    pub fn load_with_starts(path: &str, volume_db: f64, starts: &[f64]) -> Result<Self, FromFileError> {
        let sound = StaticSoundData::from_file(
            path, StaticSoundSettings::default().volume(Volume::Decibels(volume_db)))?;
        Ok(SoundAtlas {
            sound, start_positions: starts.to_vec().into_boxed_slice()
        })
    }

    pub fn load_with_stride(path: &str, volume_db: f64, stride_s: f64) -> Result<Self, FromFileError> {
        let sound = StaticSoundData::from_file(
            path, StaticSoundSettings::default().volume(Volume::Decibels(volume_db)))?;
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
