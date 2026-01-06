use std::{borrow::Cow, fs::File, io::{BufRead, BufReader, Cursor, Error, ErrorKind, Read, Seek}, path::{Path, PathBuf}, sync::Arc};
use rc_zip_sync::{EntryReader, ReadZip, ArchiveHandle};

pub trait AssetSource {
    type Reader<'a>: BufRead where Self: 'a;
    fn get_reader<'a>(&'a self, path: &Path) -> std::io::Result<Self::Reader<'a>>;
    fn get_bytes(&self, path: &Path) -> std::io::Result<Cow<'static, [u8]>>;
}

#[derive(Clone, Debug)]
pub struct LocalAssetFolder {
    pub base_path: PathBuf,
}

impl LocalAssetFolder {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let base_path: &Path = path.as_ref();
        LocalAssetFolder {
            base_path: PathBuf::from(&base_path),
        }
    }
}

impl AssetSource for LocalAssetFolder {
    type Reader<'a> = BufReader<File>;
    fn get_reader<'a>(&'a self, path: &Path) -> std::io::Result<Self::Reader<'a>> {
        let path: PathBuf = [&self.base_path, path].iter().collect();
        let file = File::open(&path)?;
        Ok(BufReader::new(file))
    }

    fn get_bytes(&self, path: &Path) -> std::io::Result<Cow<'static, [u8]>> {
        let path: PathBuf = [&self.base_path, path].iter().collect();
        let contents = std::fs::read(path)?;
        Ok(Cow::Owned(contents))
    }
}

pub struct LoadedZipBundle<'a> {
    pub archive: ArchiveHandle<'a, &'a[u8]>,
}

impl<'a> LoadedZipBundle<'a> {
    pub fn new(data: &'a &'a[u8]) -> std::io::Result<Self> {
        let archive = data.read_zip()?;
        Ok(LoadedZipBundle{archive})
    }
}

impl AssetSource for LoadedZipBundle<'_>{
    type Reader<'a> = BufReader<EntryReader<&'a [u8]>> where Self: 'a;

    fn get_reader<'a>(&'a self, path: &Path) -> std::io::Result<Self::Reader<'a>> {
        let file= self.archive.by_name(path.to_string_lossy()).ok_or(Error::from(ErrorKind::NotFound))?;
        let reader = file.reader();
        Ok(BufReader::new(reader))
    }

    fn get_bytes(&self, path: &Path) -> std::io::Result<Cow<'static, [u8]>> {
        let file= self.archive.by_name(path.to_string_lossy()).ok_or(Error::from(ErrorKind::NotFound))?;
        let bytes = file.bytes()?;
        Ok(Cow::Owned(bytes))
    }
}