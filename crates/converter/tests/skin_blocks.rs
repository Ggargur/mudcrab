//! Byte-level parsing of the Skyrim SE skin blocks: `NiSkinData`,
//! `BSDismemberSkinInstance`, the per-partition data of `NiSkinPartition`, and
//! the skin link of a legacy `NiTriShape`. The bytes are written by hand here,
//! so each layout is checked against an independent statement of it.

use project_wormhole_nif::{bs::prelude::DismemberPartition, nif_block::NifBlock};
use project_wormhole_shared::glam::{Mat3, Vec3};

/// Bethesda stream version of Skyrim Special Edition NIFs.
const SSE: u32 = 100;
/// Bethesda stream version of the original Skyrim release.
const LEGACY: u32 = 83;

/// Appends little-endian bytes of each value.
fn push_f32s(bytes: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

/// Appends little-endian `u16` values.
fn push_u16s(bytes: &mut Vec<u8>, values: &[u16]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

/// Appends a `NiTransform`: the rotation's rows, then translation, then scale.
fn push_transform(bytes: &mut Vec<u8>, rows: [[f32; 3]; 3], translation: [f32; 3], scale: f32) {
    for row in rows {
        push_f32s(bytes, &row);
    }
    push_f32s(bytes, &translation);
    push_f32s(bytes, &[scale]);
}

/// A quarter turn about +Z, written row by row as a NIF stores it.
const QUARTER_TURN_Z: [[f32; 3]; 3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Writes `NiSkinData` with two bones; the per-bone vertex weights are
/// written only when `with_weights` is set, as the format requires.
fn skin_data_bytes(with_weights: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    push_transform(&mut bytes, IDENTITY, [0.0, 0.0, -5.0], 1.0);
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.push(u8::from(with_weights));

    push_transform(&mut bytes, QUARTER_TURN_Z, [1.0, 2.0, 3.0], 2.0);
    push_f32s(&mut bytes, &[0.5, 0.25, 0.125, 7.0]); // bounding sphere
    push_u16s(&mut bytes, &[2]);
    if with_weights {
        push_u16s(&mut bytes, &[4]);
        push_f32s(&mut bytes, &[0.75]);
        push_u16s(&mut bytes, &[9]);
        push_f32s(&mut bytes, &[1.0]);
    }

    push_transform(&mut bytes, IDENTITY, [0.0, 0.0, 0.0], 1.0);
    push_f32s(&mut bytes, &[0.0, 0.0, 0.0, 1.0]);
    push_u16s(&mut bytes, &[1]);
    if with_weights {
        push_u16s(&mut bytes, &[4]);
        push_f32s(&mut bytes, &[0.25]);
    }
    bytes
}

/// Parses one block by type name and returns it, asserting every byte was consumed.
fn parse_whole(bytes: &[u8], block_type: &str, bethesda_version: u32) -> NifBlock {
    let (remaining, block) =
        NifBlock::parse_with_version(bytes, block_type.into(), bethesda_version)
            .unwrap_or_else(|error| panic!("{block_type} failed to parse: {error:?}"));
    assert!(
        remaining.is_empty(),
        "{block_type} left {} bytes unread",
        remaining.len()
    );
    block
}

/// Checks the transform convention and the per-bone weight pairs.
#[test]
fn skin_data_reads_bone_transforms_bounds_and_vertex_weights() {
    let NifBlock::NiSkinData(data) = parse_whole(&skin_data_bytes(true), "NiSkinData", SSE) else {
        panic!("NiSkinData parsed as another block");
    };
    assert!(data.has_vertex_weights);
    assert_eq!(data.skin_transform.translation, Vec3::new(0.0, 0.0, -5.0));
    assert_eq!(data.bones.len(), 2);

    let bone = &data.bones[0];
    // The rows above turn +X onto +Y, the same reading NiAVObject rotations get.
    assert_eq!(bone.transform.rotation * Vec3::X, Vec3::Y);
    assert!(
        bone.transform
            .rotation
            .abs_diff_eq(Mat3::from_rotation_z(std::f32::consts::FRAC_PI_2), 1e-6)
    );
    // Scale 2, then the quarter turn, then the translation: (1,0,0) -> (0,2,0) + (1,2,3).
    let moved = bone.transform.to_mat4().transform_point3(Vec3::X);
    assert!(
        (moved - Vec3::new(1.0, 4.0, 3.0)).length() < 1e-6,
        "{moved}"
    );
    assert_eq!(bone.bound_center, Vec3::new(0.5, 0.25, 0.125));
    assert_eq!(bone.bound_radius, 7.0);
    assert_eq!(bone.vertex_weights, vec![(4, 0.75), (9, 1.0)]);
    assert_eq!(data.bones[1].vertex_weights, vec![(4, 0.25)]);
}

/// Without the vertex-weight flag each bone still stores its vertex count, but no pairs.
#[test]
fn skin_data_without_vertex_weights_keeps_counts_but_no_pairs() {
    let NifBlock::NiSkinData(data) = parse_whole(&skin_data_bytes(false), "NiSkinData", SSE) else {
        panic!("NiSkinData parsed as another block");
    };
    assert!(!data.has_vertex_weights);
    assert!(data.bones.iter().all(|bone| bone.vertex_weights.is_empty()));
}

/// A truncated block and a bone count larger than the bytes can hold are errors, not panics.
#[test]
fn skin_data_rejects_truncated_and_oversized_input() {
    let bytes = skin_data_bytes(true);
    assert!(
        NifBlock::parse_with_version(&bytes[..bytes.len() - 3], "NiSkinData".into(), SSE).is_err()
    );

    let mut oversized = Vec::new();
    push_transform(&mut oversized, IDENTITY, [0.0; 3], 1.0);
    oversized.extend_from_slice(&u32::MAX.to_le_bytes());
    oversized.push(1);
    assert!(NifBlock::parse_with_version(&oversized, "NiSkinData".into(), SSE).is_err());
}

/// Writes the `NiSkinInstance` fields: data, partition, skeleton root, bones.
fn skin_instance_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in [7u32, 8, 0, 3, 2, 4, 6] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// `BSDismemberSkinInstance` keeps its body-part list; `NiSkinInstance` has none.
#[test]
fn dismember_skin_instance_keeps_body_parts() {
    let NifBlock::NiSkinInstance(plain) =
        parse_whole(&skin_instance_bytes(), "NiSkinInstance", SSE)
    else {
        panic!("NiSkinInstance parsed as another block");
    };
    assert_eq!(
        (plain.data, plain.skin_partition, plain.skeleton_root),
        (7, 8, 0)
    );
    assert_eq!(plain.bones, vec![2, 4, 6]);
    assert!(plain.dismember_partitions.is_none());

    let mut bytes = skin_instance_bytes();
    bytes.extend_from_slice(&2u32.to_le_bytes());
    push_u16s(&mut bytes, &[1, 32, 3, 38]);
    let block = parse_whole(&bytes, "BSDismemberSkinInstance", SSE);
    assert_eq!(format!("{block:?}"), "BSDismemberSkinInstance");
    let NifBlock::NiSkinInstance(dismember) = block else {
        panic!("BSDismemberSkinInstance parsed as another block");
    };
    assert_eq!(dismember.bones, vec![2, 4, 6]);
    assert_eq!(
        dismember.dismember_partitions,
        Some(vec![
            DismemberPartition {
                flags: 1,
                body_part: 32
            },
            DismemberPartition {
                flags: 3,
                body_part: 38
            },
        ])
    );

    let mut short = skin_instance_bytes();
    short.extend_from_slice(&2u32.to_le_bytes());
    push_u16s(&mut short, &[1, 32]);
    assert!(NifBlock::parse_with_version(&short, "BSDismemberSkinInstance".into(), SSE).is_err());
}

/// Vertex flags: position (0x1) and skin weights plus bone indices (0x40).
const POSITION_AND_SKIN: u64 = 0x1 | 0x40;
/// One vertex: position and X bitangent (16 bytes), four half weights and four bone indices (12).
const VERTEX_BYTES: usize = 28;

/// Appends one SSE partition with a vertex map, weights, bone indices and triangles.
fn push_partition(
    bytes: &mut Vec<u8>,
    palette: &[u16],
    vertex_map: &[u16],
    weights: &[f32],
    bone_indices: &[u8],
    triangles: &[[u16; 3]],
) {
    let weights_per_vertex = 2u16;
    push_u16s(
        bytes,
        &[
            vertex_map.len() as u16,
            triangles.len() as u16,
            palette.len() as u16,
            0, // strips
            weights_per_vertex,
        ],
    );
    push_u16s(bytes, palette);
    bytes.push(1);
    push_u16s(bytes, vertex_map);
    bytes.push(1);
    push_f32s(bytes, weights);
    bytes.push(1); // faces, partition-local
    for _ in triangles {
        push_u16s(bytes, &[0, 0, 0]);
    }
    bytes.push(1);
    bytes.extend_from_slice(bone_indices);
    bytes.push(0); // LOD level
    bytes.push(1); // global vertex buffer
    bytes.extend_from_slice(&vertex_desc().to_le_bytes());
    for triangle in triangles {
        push_u16s(bytes, triangle);
    }
}

/// 0.5 as an IEEE 754 half-precision float.
const HALF_ONE_HALF: u16 = 0x3800;

/// The descriptor for `POSITION_AND_SKIN` vertices.
fn vertex_desc() -> u64 {
    (VERTEX_BYTES as u64 / 4) | (POSITION_AND_SKIN << 44)
}

/// Writes an SSE `NiSkinPartition` with four shared vertices and two partitions
/// whose palettes are not in skin-instance order.
fn sse_partition_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&((4 * VERTEX_BYTES) as u32).to_le_bytes());
    bytes.extend_from_slice(&(VERTEX_BYTES as u32).to_le_bytes());
    bytes.extend_from_slice(&vertex_desc().to_le_bytes());
    for vertex in 0..4u8 {
        push_f32s(&mut bytes, &[f32::from(vertex), 0.0, 0.0, 0.0]);
        push_u16s(&mut bytes, &[HALF_ONE_HALF, HALF_ONE_HALF, 0, 0]);
        bytes.extend_from_slice(&[vertex, 1, 0, 0]);
    }
    push_partition(
        &mut bytes,
        &[2, 0],
        &[0, 1, 2],
        &[1.0, 0.0, 0.25, 0.75, 0.5, 0.5],
        &[0, 1, 1, 0, 0, 1],
        &[[0, 1, 2]],
    );
    push_partition(
        &mut bytes,
        &[1],
        &[1, 2, 3],
        &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        &[0, 0, 0, 0, 0, 0],
        &[[1, 2, 3], [3, 2, 1]],
    );
    bytes
}

/// Each partition keeps its palette, vertex map, weights, bone indices and
/// triangles, while the concatenated triangles and the shared vertex buffer
/// the static path reads stay as before.
#[test]
fn sse_skin_partition_keeps_per_partition_influences() {
    let NifBlock::NiSkinPartition(partition) =
        parse_whole(&sse_partition_bytes(), "NiSkinPartition", SSE)
    else {
        panic!("NiSkinPartition parsed as another block");
    };
    assert_eq!(partition.vertex_data.len(), 4);
    assert_eq!(
        partition.vertex_data[3]
            .bone_indices
            .map(|indices| indices.to_array()),
        Some([3, 1, 0, 0])
    );
    assert_eq!(partition.partitions.len(), 2);

    let first = &partition.partitions[0];
    assert_eq!(first.weights_per_vertex, 2);
    assert_eq!(first.bones, vec![2, 0]);
    assert_eq!(first.vertex_map, vec![0, 1, 2]);
    assert_eq!(first.vertex_weights, vec![1.0, 0.0, 0.25, 0.75, 0.5, 0.5]);
    assert_eq!(first.bone_indices, vec![0, 1, 1, 0, 0, 1]);
    assert_eq!(first.triangles.len(), 1);

    let second = &partition.partitions[1];
    assert_eq!(second.bones, vec![1]);
    assert_eq!(second.vertex_map, vec![1, 2, 3]);
    assert_eq!(second.triangles.len(), 2);

    let concatenated: Vec<_> = partition
        .partitions
        .iter()
        .flat_map(|part| part.triangles.iter().copied())
        .collect();
    assert_eq!(partition.triangles, concatenated);
}

/// The legacy layout still recovers triangles only, with no per-partition data.
#[test]
fn legacy_skin_partition_has_no_per_partition_data() {
    let mut bytes = 1u32.to_le_bytes().to_vec();
    push_u16s(&mut bytes, &[3, 1, 1, 0, 1]); // vertices, triangles, bones, strips, weights per vertex
    push_u16s(&mut bytes, &[0]); // palette
    bytes.push(0); // no vertex map
    bytes.push(0); // no weights
    bytes.push(1); // faces
    push_u16s(&mut bytes, &[0, 1, 2]);
    bytes.push(0); // no bone indices
    push_u16s(&mut bytes, &[0]); // Bethesda partition flags
    let NifBlock::NiSkinPartition(partition) = parse_whole(&bytes, "NiSkinPartition", LEGACY)
    else {
        panic!("NiSkinPartition parsed as another block");
    };
    assert_eq!(partition.triangles.len(), 1);
    assert!(partition.partitions.is_empty());
}

/// A legacy `NiTriShape` keeps its skin-instance link instead of discarding it.
#[test]
fn legacy_tri_shape_keeps_its_skin_instance_link() {
    let mut bytes = Vec::new();
    for value in [0u32, 0, u32::MAX, 14] {
        bytes.extend_from_slice(&value.to_le_bytes()); // name, extra data, controller, flags
    }
    push_f32s(&mut bytes, &[0.0; 3]);
    for row in IDENTITY {
        push_f32s(&mut bytes, &row);
    }
    push_f32s(&mut bytes, &[1.0]);
    for value in [u32::MAX, 5, 6, 0, 0] {
        bytes.extend_from_slice(&value.to_le_bytes()); // collision, data, skin, materials, active material
    }
    bytes.push(0); // material needs update
    for value in [9u32, u32::MAX] {
        bytes.extend_from_slice(&value.to_le_bytes()); // shader, alpha
    }
    let NifBlock::NiTriShape(shape) = parse_whole(&bytes, "NiTriShape", LEGACY) else {
        panic!("NiTriShape parsed as another block");
    };
    assert_eq!((shape.data, shape.skin_instance), (5, 6));
    assert_eq!(shape.shader_property, 9);
}
