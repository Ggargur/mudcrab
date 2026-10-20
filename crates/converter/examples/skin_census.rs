//! Census of the Skyrim SE skins in the NIFs behind a converted pack (issue #273).
//!
//! `cargo run --release -p converter --example skin_census -- <converted-pack> <ingestion-cache> [report.json]`
//!
//! `<converted-pack>` is a converter output folder: its `conversion-manifest.json` names every
//! source NIF and its SHA-256, and its `skyrim_world.db` names the models placed in the world.
//! `<ingestion-cache>` is the folder holding `sha256/<xx>/<hash>` blobs (the converter's
//! `.ingestion-cache`). Every NIF is read from its blob, so nothing is extracted. Read only.
//!
//! The report is printed as JSON (and written to `report.json` when given), once for every
//! source NIF and once for the subset whose model is placed by a world reference. It answers
//! what the skin exporter must decide before it ships:
//! - which shape and skin-instance layouts occur, and how often links are missing or wrong;
//! - whether the bone indices in the shared vertex buffer name the skin instance's bones
//!   ("global") or the partition palette ("local"), checked per vertex against the partition's
//!   own weight and palette rows, and against `NiSkinData`'s per-bone weights when present;
//! - which weights would be rejected (NaN, negative, zero sum, out-of-range bone);
//! - how far `Σ w · bone_to_root · bone_data · v` lands from `shape_to_root · v`, which is
//!   zero when the skin's bind pose matches the node hierarchy the static output uses.

use converter::mesh::MeshConverter;
use project_wormhole_nif::{
    bs::prelude::{BSTriShape, NiSkinData, NiSkinInstance, NiSkinPartition},
    nif_block::{NiAVObject, NifBlock},
    nif_file::NifFile,
};
use project_wormhole_shared::glam::{Mat4, Vec3};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
};

/// A missing block link.
const NULL_REF: u32 = u32::MAX;
/// Bevy 0.19's joint limit per skinned mesh.
const MAX_JOINTS: usize = 256;
/// Sample paths kept per counted case.
const SAMPLES: usize = 5;
/// Half-float weights are compared with this tolerance.
const WEIGHT_TOLERANCE: f32 = 2e-3;
/// Skin and node transforms are compared with this tolerance per matrix element.
const TRANSFORM_TOLERANCE: f32 = 1e-3;

/// A count with up to `SAMPLES` example paths.
#[derive(Default, Serialize)]
struct Tally {
    count: usize,
    samples: Vec<String>,
}

impl Tally {
    /// Counts one occurrence, keeping the path while there is room for samples.
    fn add(&mut self, path: &str) {
        self.count += 1;
        if self.samples.len() < SAMPLES && !self.samples.iter().any(|sample| sample == path) {
            self.samples.push(path.to_owned());
        }
    }

    /// Adds another tally's count and samples.
    fn merge(&mut self, other: &Tally) {
        self.count += other.count;
        for sample in &other.samples {
            if self.samples.len() < SAMPLES && !self.samples.contains(sample) {
                self.samples.push(sample.clone());
            }
        }
    }
}

/// Per-vertex agreement counts for one comparison.
#[derive(Default, Serialize, Clone, Copy)]
struct Agreement {
    compared: usize,
    agree: usize,
}

impl Agreement {
    /// Records one comparison.
    fn record(&mut self, agrees: bool) {
        self.compared += 1;
        self.agree += usize::from(agrees);
    }

    /// Adds another agreement's counts.
    fn merge(&mut self, other: Agreement) {
        self.compared += other.compared;
        self.agree += other.agree;
    }
}

#[derive(Default, Serialize)]
struct Census {
    files: usize,
    /// NIFs whose blob is not in the ingestion cache, so they were not read.
    missing_blob: Tally,
    unreadable: Tally,
    skinned_files: usize,
    /// Skin blocks the parser could not read, by type; each one loses its skin today.
    skin_block_fallbacks: BTreeMap<String, Tally>,
    /// `<shape type> -> <link type>` for every shape with a non-null skin link.
    layouts: BTreeMap<String, Tally>,
    /// Problems that would make the exporter reject the shape, by kind.
    rejections: BTreeMap<String, Tally>,
    shapes_with_vertex_weights_in_skin_data: usize,
    shapes_without_vertex_weights_in_skin_data: usize,
    max_bones: usize,
    /// Vertices whose buffer influences match the partition's under each reading of the buffer's bone indices.
    buffer_indices_as_instance_bones: Agreement,
    buffer_indices_as_partition_palette: Agreement,
    /// Vertices whose partition influences match `NiSkinData`'s per-bone weights.
    partition_matches_skin_data: Agreement,
    /// Shapes with at least one vertex where the instance-bone reading of the buffer disagrees.
    index_disagreement_shapes: Tally,
    /// Shapes with at least one vertex where the partition disagrees with `NiSkinData`.
    skin_data_disagreement_shapes: Tally,
    /// Shapes whose overall skin transform is the inverse of the shape's transform relative to the skeleton root.
    skin_transform_inverts_shape_to_skeleton_root: Agreement,
    /// Shapes whose bind deviation could be measured, by whether the file animates transforms
    /// (it has controller or interpolator blocks) and bucket of the largest vertex deviation.
    bind_deviation: BTreeMap<String, Tally>,
    bind_unmeasured: Tally,
}

impl Census {
    /// Adds another census's counts.
    fn merge(&mut self, other: &Census) {
        self.files += other.files;
        self.missing_blob.merge(&other.missing_blob);
        self.unreadable.merge(&other.unreadable);
        self.skinned_files += other.skinned_files;
        merge_tallies(&mut self.skin_block_fallbacks, &other.skin_block_fallbacks);
        merge_tallies(&mut self.layouts, &other.layouts);
        merge_tallies(&mut self.rejections, &other.rejections);
        self.shapes_with_vertex_weights_in_skin_data +=
            other.shapes_with_vertex_weights_in_skin_data;
        self.shapes_without_vertex_weights_in_skin_data +=
            other.shapes_without_vertex_weights_in_skin_data;
        self.max_bones = self.max_bones.max(other.max_bones);
        self.buffer_indices_as_instance_bones
            .merge(other.buffer_indices_as_instance_bones);
        self.buffer_indices_as_partition_palette
            .merge(other.buffer_indices_as_partition_palette);
        self.partition_matches_skin_data
            .merge(other.partition_matches_skin_data);
        self.index_disagreement_shapes
            .merge(&other.index_disagreement_shapes);
        self.skin_data_disagreement_shapes
            .merge(&other.skin_data_disagreement_shapes);
        self.skin_transform_inverts_shape_to_skeleton_root
            .merge(other.skin_transform_inverts_shape_to_skeleton_root);
        merge_tallies(&mut self.bind_deviation, &other.bind_deviation);
        self.bind_unmeasured.merge(&other.bind_unmeasured);
    }

    /// Counts a reason the exporter would reject a shape.
    fn reject(&mut self, kind: &str, path: &str) {
        self.rejections
            .entry(kind.to_owned())
            .or_default()
            .add(path);
    }
}

/// Adds every tally of `other` into `into`.
fn merge_tallies<K: Ord + Clone>(into: &mut BTreeMap<K, Tally>, other: &BTreeMap<K, Tally>) {
    for (key, tally) in other {
        into.entry(key.clone()).or_default().merge(tally);
    }
}

/// The BSTriShape payload of every SSE shape block type, with the type's name.
fn sse_shape(block: &NifBlock) -> Option<(&'static str, &BSTriShape)> {
    match block {
        NifBlock::BSTriShape(shape) => Some(("BSTriShape", shape)),
        NifBlock::BSDynamicTriShape(shape) => Some(("BSDynamicTriShape", &shape.bs_tri_shape)),
        NifBlock::BSSubIndexTriShape(shape) => Some(("BSSubIndexTriShape", &shape.bs_tri_shape)),
        NifBlock::BSLODTriShape(shape) => Some(("BSLODTriShape", &shape.bs_tri_shape)),
        _ => None,
    }
}

/// The scene-graph object fields of a node or SSE shape block.
fn av_object(block: &NifBlock) -> Option<&NiAVObject> {
    match block {
        NifBlock::NiNode(node) | NifBlock::BSFadeNode(node) => Some(&node.av),
        NifBlock::BSMultiBoundNode(node) => Some(&node.node.av),
        _ => sse_shape(block).map(|(_, shape)| &shape.av),
    }
}

/// A node's children, for the node types the static scene keeps.
fn children(block: &NifBlock) -> &[u32] {
    match block {
        NifBlock::NiNode(node) | NifBlock::BSFadeNode(node) => &node.children,
        NifBlock::BSMultiBoundNode(node) => &node.node.children,
        _ => &[],
    }
}

/// Root-relative transforms, composed the way the static scene composes node transforms.
struct Hierarchy<'a> {
    nif: &'a NifFile,
    parents: HashMap<u32, u32>,
}

impl<'a> Hierarchy<'a> {
    /// Records each block's parent from the node child lists.
    fn new(nif: &'a NifFile) -> Self {
        let mut parents = HashMap::new();
        for (index, block) in nif.blocks.iter().enumerate() {
            for &child in children(block) {
                parents.entry(child).or_insert(index as u32);
            }
        }
        Self { nif, parents }
    }

    /// The block's own transform: scale, then rotation, then translation.
    fn local(&self, block: u32) -> Option<Mat4> {
        let av = av_object(self.nif.blocks.get(block as usize)?)?;
        Some(
            Mat4::from_translation(av.translation.0.0)
                * Mat4::from_mat3(av.rotation.0.0)
                * Mat4::from_scale(Vec3::splat(av.scale.0)),
        )
    }

    /// The transform from the block's space to the file root's; `None` when a block on the
    /// way has no readable transform, the parent links loop, or the chain stops short of the
    /// root block because a parent is a node type the parser does not read.
    fn to_root(&self, block: u32) -> Option<Mat4> {
        let mut transform = self.local(block)?;
        let mut current = block;
        for _ in 0..self.nif.blocks.len() {
            let Some(&parent) = self.parents.get(&current) else {
                return (current == 0).then_some(transform);
            };
            transform = self.local(parent)? * transform;
            current = parent;
        }
        None
    }
}

/// Whether two matrices agree element by element within `TRANSFORM_TOLERANCE`.
fn matrices_agree(a: Mat4, b: Mat4) -> bool {
    a.to_cols_array()
        .iter()
        .zip(b.to_cols_array())
        .all(|(x, y)| (x - y).abs() <= TRANSFORM_TOLERANCE)
}

/// Sorted `(bone, weight)` pairs with weight above zero.
fn influences(pairs: impl IntoIterator<Item = (u32, f32)>) -> Vec<(u32, f32)> {
    let mut pairs: Vec<_> = pairs
        .into_iter()
        .filter(|(_, weight)| *weight > 0.0)
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    pairs
}

/// Whether two influence lists name the same bones with the same weights.
fn same_influences(a: &[(u32, f32)], b: &[(u32, f32)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.0 == y.0 && (x.1 - y.1).abs() <= WEIGHT_TOLERANCE)
}

/// A skinned SSE shape whose links all resolved.
struct ResolvedShape<'a> {
    block: u32,
    shape: &'a BSTriShape,
    instance: &'a NiSkinInstance,
    data: &'a NiSkinData,
    partition: &'a NiSkinPartition,
}

/// Follows a shape's skin links, counting the first broken one as a rejection.
fn resolve<'a>(
    nif: &'a NifFile,
    block: u32,
    shape: &'a BSTriShape,
    path: &str,
    census: &mut Census,
) -> Option<ResolvedShape<'a>> {
    let instance = match nif.blocks.get(shape.skin as usize) {
        Some(NifBlock::NiSkinInstance(instance)) => instance,
        Some(NifBlock::BSSkinInstance(_)) => return None,
        Some(NifBlock::Unhandled) => {
            census.reject("skin instance unparsed", path);
            return None;
        }
        Some(_) => {
            census.reject("skin link is not a skin instance", path);
            return None;
        }
        None => {
            census.reject("skin link out of range", path);
            return None;
        }
    };
    let data = match nif.blocks.get(instance.data as usize) {
        Some(NifBlock::NiSkinData(data)) => data,
        Some(NifBlock::Unhandled) => {
            census.reject("skin data unparsed", path);
            return None;
        }
        Some(_) => {
            census.reject("skin data link is not NiSkinData", path);
            return None;
        }
        None => {
            census.reject("skin data link missing", path);
            return None;
        }
    };
    let partition = match nif.blocks.get(instance.skin_partition as usize) {
        Some(NifBlock::NiSkinPartition(partition)) => partition,
        Some(NifBlock::Unhandled) => {
            census.reject("skin partition unparsed", path);
            return None;
        }
        Some(_) => {
            census.reject("skin partition link is not NiSkinPartition", path);
            return None;
        }
        None => {
            census.reject("skin partition link missing", path);
            return None;
        }
    };
    Some(ResolvedShape {
        block,
        shape,
        instance,
        data,
        partition,
    })
}

/// Checks one resolved shape's bones, weights, index convention and bind pose.
fn measure(
    nif: &NifFile,
    hierarchy: &Hierarchy,
    skin: &ResolvedShape,
    animated: bool,
    path: &str,
    census: &mut Census,
) {
    // Each rejection kind counts once per shape, however many vertices show it.
    let mut kinds = BTreeSet::new();
    let bones = &skin.instance.bones;
    census.max_bones = census.max_bones.max(bones.len());
    if bones.len() > MAX_JOINTS {
        kinds.insert("more bones than Bevy's joint limit");
    }
    if bones.len() != skin.data.bones.len() {
        kinds.insert("NiSkinData bone count differs from the instance");
    }
    if bones.iter().collect::<BTreeSet<_>>().len() != bones.len() {
        kinds.insert("duplicate bone");
    }
    if bones.iter().any(|&bone| {
        av_object(
            nif.blocks
                .get(bone as usize)
                .unwrap_or(&NifBlock::Unhandled),
        )
        .is_none()
    }) {
        kinds.insert("bone is not a readable node");
    }
    if let Some(dismember) = &skin.instance.dismember_partitions
        && dismember.len() != skin.partition.partitions.len()
    {
        kinds.insert("dismember body parts differ from partitions");
    }
    if skin.data.has_vertex_weights {
        census.shapes_with_vertex_weights_in_skin_data += 1;
    } else {
        census.shapes_without_vertex_weights_in_skin_data += 1;
    }
    if !skin.shape.vertex_desc.has_skin_data() && !skin.partition.vertex_desc.has_skin_data() {
        kinds.insert("vertex buffer has no skin data");
    }

    // Influences per shared-buffer vertex as the partitions state them, in instance-bone terms.
    let vertex_count = skin.partition.vertex_data.len();
    let mut from_partitions: Vec<Option<Vec<(u32, f32)>>> = vec![None; vertex_count];
    for part in &skin.partition.partitions {
        let per_vertex = usize::from(part.weights_per_vertex);
        for (row, &vertex) in part.vertex_map.iter().enumerate() {
            let vertex = usize::from(vertex);
            if vertex >= vertex_count || from_partitions[vertex].is_some() {
                continue;
            }
            let range = row * per_vertex..(row + 1) * per_vertex;
            let (Some(weights), Some(indices)) = (
                part.vertex_weights.get(range.clone()),
                part.bone_indices.get(range),
            ) else {
                continue;
            };
            let mut pairs = Vec::new();
            for (&weight, &local) in weights.iter().zip(indices) {
                if weight > 0.0 {
                    match part.bones.get(usize::from(local)) {
                        Some(&bone) => pairs.push((u32::from(bone), weight)),
                        None => {
                            kinds.insert("partition palette index out of range");
                        }
                    }
                }
            }
            from_partitions[vertex] = Some(influences(pairs));
        }
    }

    let mut skin_data_weights: HashMap<(u32, u16), f32> = HashMap::new();
    for (bone, data) in skin.data.bones.iter().enumerate() {
        for &(vertex, weight) in &data.vertex_weights {
            skin_data_weights.insert((bone as u32, vertex), weight);
        }
    }

    let mut shape_indices = Agreement::default();
    let mut shape_skin_data = Agreement::default();
    for (vertex, data) in skin.partition.vertex_data.iter().enumerate() {
        let (Some(weights), Some(indices)) = (data.bone_weights, data.bone_indices) else {
            continue;
        };
        let weights = weights.to_array();
        let indices = indices.to_array();
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            kinds.insert("NaN or negative weight");
        } else if weights.iter().sum::<f32>() <= 0.0 {
            kinds.insert("vertex weights sum to zero");
        }
        if indices
            .iter()
            .zip(weights)
            .any(|(&index, weight)| weight > 0.0 && usize::from(index) >= bones.len())
        {
            kinds.insert("weighted bone index past the instance's bones");
        }
        let Some(stated) = &from_partitions[vertex] else {
            continue;
        };
        let as_instance = influences(
            indices
                .iter()
                .zip(weights)
                .map(|(&index, weight)| (u32::from(index), weight)),
        );
        shape_indices.record(same_influences(&as_instance, stated));
        // The palette reading needs to know which partition the vertex belongs to.
        if let Some(part) = skin
            .partition
            .partitions
            .iter()
            .find(|part| part.vertex_map.contains(&(vertex as u16)))
        {
            let as_palette =
                influences(indices.iter().zip(weights).filter_map(|(&index, weight)| {
                    part.bones
                        .get(usize::from(index))
                        .map(|&bone| (u32::from(bone), weight))
                }));
            census
                .buffer_indices_as_partition_palette
                .record(same_influences(&as_palette, stated));
        }
        if skin.data.has_vertex_weights {
            shape_skin_data.record(stated.iter().all(|&(bone, weight)| {
                skin_data_weights
                    .get(&(bone, vertex as u16))
                    .is_some_and(|stored| (stored - weight).abs() <= WEIGHT_TOLERANCE)
            }));
        }
    }
    census.buffer_indices_as_instance_bones.merge(shape_indices);
    census.partition_matches_skin_data.merge(shape_skin_data);
    if shape_indices.agree < shape_indices.compared {
        census.index_disagreement_shapes.add(path);
    }
    if shape_skin_data.agree < shape_skin_data.compared {
        census.skin_data_disagreement_shapes.add(path);
    }

    for kind in kinds {
        census.reject(kind, path);
    }

    // Overall skin transform against the shape's transform relative to the skeleton root.
    if let (Some(shape_to_root), Some(skeleton_to_root)) = (
        hierarchy.to_root(skin.block),
        hierarchy.to_root(skin.instance.skeleton_root),
    ) {
        let shape_to_skeleton = skeleton_to_root.inverse() * shape_to_root;
        census
            .skin_transform_inverts_shape_to_skeleton_root
            .record(matrices_agree(
                skin.data.skin_transform.to_mat4(),
                shape_to_skeleton.inverse(),
            ));
    }

    // Bind pose: Σ w · bone_to_root · bone_data · v against shape_to_root · v.
    let Some(shape_to_root) = hierarchy.to_root(skin.block) else {
        census.bind_unmeasured.add(path);
        return;
    };
    let Some(bone_matrices) = bones
        .iter()
        .zip(&skin.data.bones)
        .map(|(&bone, data)| Some(hierarchy.to_root(bone)? * data.transform.to_mat4()))
        .collect::<Option<Vec<_>>>()
    else {
        census.bind_unmeasured.add(path);
        return;
    };
    let mut worst = 0.0f32;
    for (vertex, data) in skin.partition.vertex_data.iter().enumerate() {
        let (Some(position), Some(stated)) = (data.position, &from_partitions[vertex]) else {
            continue;
        };
        let total: f32 = stated.iter().map(|(_, weight)| weight).sum();
        if total <= 0.0 {
            continue;
        }
        let skinned = stated.iter().fold(Vec3::ZERO, |sum, &(bone, weight)| {
            bone_matrices.get(bone as usize).map_or(sum, |matrix| {
                sum + matrix.transform_point3(position) * (weight / total)
            })
        });
        worst = worst.max((skinned - shape_to_root.transform_point3(position)).length());
    }
    let bucket = match worst {
        d if d < 1e-3 => "a: < 0.001",
        d if d < 1e-2 => "b: < 0.01",
        d if d < 0.1 => "c: < 0.1",
        d if d < 1.0 => "d: < 1",
        d if d < 10.0 => "e: < 10",
        _ => "f: >= 10",
    };
    let kind = if animated { "animated" } else { "not animated" };
    census
        .bind_deviation
        .entry(format!("{kind} / {bucket}"))
        .or_default()
        .add(path);
}

/// Counts one NIF's skin layouts and measures each SSE skin.
fn analyze(nif_path: &Path, path: &str) -> Census {
    let mut census = Census {
        files: 1,
        ..Census::default()
    };
    let (nif, diagnostics) = match MeshConverter::open_nif(nif_path) {
        Ok(opened) => opened,
        Err(_) => {
            census.unreadable.add(path);
            return census;
        }
    };
    for (block_type, _) in diagnostics
        .fallback_blocks
        .iter()
        .filter(|(block_type, _)| {
            matches!(
                block_type.as_str(),
                "NiSkinInstance" | "BSDismemberSkinInstance" | "NiSkinData" | "NiSkinPartition"
            )
        })
    {
        census
            .skin_block_fallbacks
            .entry(block_type.clone())
            .or_default()
            .add(path);
    }
    let animated = diagnostics
        .block_types
        .keys()
        .any(|block_type| block_type.contains("Controller") || block_type.contains("Interpolator"));
    let hierarchy = Hierarchy::new(&nif);
    let mut skinned = false;
    for (index, block) in nif.blocks.iter().enumerate() {
        let link_type = |link: u32| match nif.blocks.get(link as usize) {
            Some(block) => format!("{block:?}"),
            None => "out of range".to_owned(),
        };
        if let NifBlock::NiTriShape(shape) = block {
            if shape.skin_instance != NULL_REF {
                skinned = true;
                let layout = format!("NiTriShape -> {}", link_type(shape.skin_instance));
                census.layouts.entry(layout).or_default().add(path);
                census.reject("skinned NiTriShape", path);
            }
            continue;
        }
        let Some((shape_type, shape)) = sse_shape(block) else {
            continue;
        };
        if shape.skin == NULL_REF {
            continue;
        }
        skinned = true;
        let layout = format!("{shape_type} -> {}", link_type(shape.skin));
        census.layouts.entry(layout).or_default().add(path);
        if let Some(skin) = resolve(&nif, index as u32, shape, path, &mut census) {
            measure(&nif, &hierarchy, &skin, animated, path, &mut census);
        }
    }
    census.skinned_files = usize::from(skinned);
    census
}

/// The manifest key of a model path as a base record stores it.
fn manifest_key(model_path: &str) -> String {
    let path = model_path.replace('\\', "/").to_ascii_lowercase();
    let path = path.strip_prefix("meshes/").unwrap_or(&path);
    format!("meshes/{path}")
}

/// Model paths of base records that at least one world reference places.
fn placed_models(database: &Path) -> BTreeSet<String> {
    let connection =
        rusqlite::Connection::open_with_flags(database, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .expect("open the world database");
    let mut statement = connection
        .prepare(
            "SELECT DISTINCT s.model_path FROM statics s \
             JOIN \"references\" r ON r.base_form_id = s.id \
             WHERE s.model_path IS NOT NULL AND s.model_path != ''",
        )
        .expect("query placed models");
    statement
        .query_map([], |row| row.get::<_, String>(0))
        .expect("read placed models")
        .map(|path| manifest_key(&path.expect("model path")))
        .collect()
}

#[derive(Serialize)]
struct Report {
    all: Census,
    placed_in_world: Census,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: skin_census <converted-pack> <ingestion-cache> [report.json]";
    let pack = PathBuf::from(args.next().expect(usage));
    let cache = PathBuf::from(args.next().expect(usage));
    let output = args.next().map(PathBuf::from);

    let manifest: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(pack.join("conversion-manifest.json")).expect("read the manifest"),
    )
    .expect("parse the manifest");
    let entries = manifest["entries"]
        .as_object()
        .expect("the manifest has an entries map");
    let placed = placed_models(&pack.join("skyrim_world.db"));

    let mut report = Report {
        all: Census::default(),
        placed_in_world: Census::default(),
    };
    let nifs: Vec<_> = entries
        .iter()
        .filter(|(path, _)| path.ends_with(".nif"))
        .filter_map(|(path, entry)| Some((path, entry["source_hash"].as_str()?)))
        .collect();
    for (done, (path, hash)) in nifs.iter().enumerate() {
        if done % 2000 == 0 {
            eprintln!("{done}/{} NIFs", nifs.len());
        }
        // A skinned mesh's key also names its skeleton: `<nif hash>:<skeleton hash>`.
        let hash = hash.split(':').next().unwrap_or_default();
        let blob = cache
            .join("sha256")
            .join(hash.get(..2).unwrap_or_default())
            .join(hash);
        let census = if blob.is_file() {
            analyze(&blob, path)
        } else {
            let mut census = Census {
                files: 1,
                ..Census::default()
            };
            census.missing_blob.add(path);
            census
        };
        report.all.merge(&census);
        if placed.contains(path.as_str()) {
            report.placed_in_world.merge(&census);
        }
    }

    let json = serde_json::to_string_pretty(&report).expect("serialize the report");
    if let Some(output) = output {
        fs::write(&output, &json).expect("write the report");
    }
    println!("{json}");
}
