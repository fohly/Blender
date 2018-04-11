# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8 compliant>

import sys
import os


# may split this out into a new file
def replace_bpy_app_version():
    """ So MD5's are predictable from output which uses blenders versions.
    """

    import bpy

    app = bpy.app
    app_fake = type(bpy)("bpy.app")

    for attr in dir(app):
        if not attr.startswith("_"):
            setattr(app_fake, attr, getattr(app, attr))

    app_fake.version = 0, 0, 0
    app_fake.version_string = "0.00 (sub 0)"
    bpy.app = app_fake


def clear_startup_blend():
    import bpy

    for scene in bpy.data.scenes:
        for obj in scene.objects:
            scene.objects.unlink(obj)


def blend_to_md5():
    import bpy
    import numpy as np
    import time
    scene = bpy.context.scene
    ROUND = 4
    useRound = False

    def floatseq2str(seq):
        return "".join([str(round(val, ROUND)) for val in seq]).encode('ASCII')

    def matrix2str(matrix):
        return floatseq2str(axis for vector in matrix for axis in vector)

    def coords2str(seq, attr):
        return floatseq2str(axis for vertex in seq for axis in getattr(vertex, attr))

    #returns an attribute from items in a sequence as a numpy array
    #faster than list comprehension
    def seq2numpyarray(seq, attribute, dtype, length_multiplier):
        data = np.empty(len(seq) * length_multiplier, dtype=dtype)
        seq.foreach_get(attribute, data)
        return data

    def hash_sequence(hash_update, seq, attribute, dtype, length_multiplier):
        data = np.empty(len(seq) * length_multiplier, dtype=dtype)
        seq.foreach_get(attribute, data)
        if (sys.byteorder != 'little'):
            data.byteswap(True)
        hash_update(data)
        del data

    #hashes the mesh data
    #supported:
    # vertices, faces, edges
    # uvs, vertex colors
    #not supported:
    # shapekeys, ...?
    # vertex groups (don't know how to access efficiently)
    # material indices
    #
    # also, the hash is order-dependent, could be resolved by sorting the data first and remapping indices
    def mesh_hash(hash_update, mesh):
        #vertex coordinates
        hash_sequence(hash_update, mesh.vertices, 'co', np.float32, 3)

        #faces
        #we only check the loops, and the loop indices in the faces
        hash_sequence(hash_update, mesh.polygons, 'loop_start', np.uint32, 1)
        hash_sequence(hash_update, mesh.polygons, 'loop_total', np.uint32, 1)
        hash_sequence(hash_update, mesh.loops, 'vertex_index', np.uint32, 1)

        #edges
        hash_sequence(hash_update, mesh.edges, 'vertices', np.uint32, 2)

        #vertex colors
        for vcolor in mesh.vertex_colors:
            hash_sequence(hash_update, vcolor.data, 'color', np.float32, 4) # vertex colors have 4 components. getting as 8 bit integer will not work

        #uvs
        for uv in mesh.uv_layers:
            hash_sequence(hash_update, uv.data, 'uv', np.float32, 2)

        #vertex groups
        #TODO

    #rotates vertices inside a polygon, so that the vertex with the lowest index is first
    #then sorts the polys based on a hash based on its vertex indices
    #returns a tuple containing:
    # - a mapping from new to old loop index (to remap per loop data, like uvs)
    # - a mapping from new to old polygon index (can be used to remap per face data)
    def rotate_sort_polys(loops, polys, polys_len):
        loops = loops.tolist() # we don't use numpy stuff in this function, and lists provide faster indexing
        polys = polys.tolist()
        polys_len = polys_len.tolist()

        # rotate verts inside poly and compute hash
        loops_map = [0] * len(loops)
        polys_hash = [0] * len(polys) # hopefully unique value per poly
        for ii in range(len(polys)):
            start_idx = polys[ii]
            length = polys_len[ii]
            min_vertex = loops[start_idx]
            min_idx = 0
            for jj in range(1, length):
                if loops[jj + start_idx] < min_vertex:
                    min_vertex = loops[jj + start_idx]
                    min_idx = jj
            poly_hash = 14695981039346656037
            for jj in range(length):
                idx = start_idx + ((jj + min_idx) % length)
                loops_map[jj + start_idx] = idx
                poly_hash = (poly_hash * 1099511628211) ^ loops[idx] # Fowler–Noll–Vo hash, except that we work per integer, not per byte
            polys_hash[ii] = poly_hash

        polys_map = np.argsort(polys_hash, kind='mergesort') # mergesort, even though we should not get collisions

        # reorder the polygons based on hash

        loops_sorted_map = [0] * len(loops)
        next_start = 0
        for ii in range(len(polys)):
            poly_idx = polys_map[ii]
            loop_start = polys[poly_idx]
            length = polys_len[poly_idx]
            for jj in range(length):
                loops_sorted_map[next_start + jj] = loops_map[loop_start + jj]
            next_start += length

        return (loops_sorted_map, polys_map)

    def hash_mapped(hash_update, data, mapping, reinterpret_type):
        data2 = data.view(dtype=reinterpret_type)
        data_sorted = data2[mapping]
        if (sys.byteorder != 'little'):
            data_sorted.byteswap(True)
        hash_update(data_sorted)
        del data_sorted

    #hashes the mesh data
    #first sorts the data to be deterministic
    #may have non-deterministic results with double vertices / double faces / etc
    #hashes:
    # - vertices - polys - edges
    # - vgroups - vcolors - uvs
    # - sharp edges
    # - material assignments
    def mesh_hash_sorted(hash_update, obj):
        mesh = obj.data

        verts_raw = seq2numpyarray(mesh.vertices, 'co', np.float32, 3)
        if (useRound):
                np.around(verts_raw, decimals=ROUND, out=verts_raw)

        verts = verts_raw.view(dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
        verts_map = np.argsort(verts, kind='mergesort', order=['x', 'y', 'z']) # mergesort, because it is stable (quicksort is based on random numbers usually)
        #verts_map maps from new index to old index. so verts_map[0] is the index inside verts of the "lowest" vertex

        verts_map_inv = np.argsort(verts_map) # maps from old to new vertex index
        verts_sorted = verts[verts_map]

        polys = seq2numpyarray(mesh.polygons, 'loop_start', np.uint32, 1)
        polys_len = seq2numpyarray(mesh.polygons, 'loop_total', np.uint32, 1)

        # prepare loop data
        # contains the vertex index of every polygon corner
        loops = seq2numpyarray(mesh.loops, 'vertex_index', np.uint32, 1) # vertex indices per loop
        loops = np.array(verts_map_inv[loops], dtype=np.uint32) # remap vertex indices of loops

        (loops_map, polys_map) = rotate_sort_polys(loops, polys, polys_len)
        loops_sorted = loops[loops_map]

        #edges
        edges_raw = seq2numpyarray(mesh.edges, 'vertices', np.uint32, 2)
        edges = np.array(verts_map_inv[edges_raw], dtype=np.uint32) # remap vertex indices
        edges_a = edges[0::2] # split into first and second vertex per edge
        edges_b = edges[1::2]
        edges_flag = edges_a < edges_b
        edges_max = np.choose(edges_flag, (edges_a, edges_b))
        edges_min = np.choose(edges_flag, (edges_b, edges_a))
        np.concatenate((edges_min, edges_max), out=edges)
        edges = edges.reshape(2, -1).transpose().reshape(-1) # put edge vertices in pairs again and flatten
        edges = edges.view(dtype=[('a', np.uint32), ('b', np.uint32)])
        edges_map = np.argsort(edges, order=['a', 'b'])
        edges_sorted = edges[edges_map]

        #uvs
        uv_names = [uv.name for uv in mesh.uv_layers]
        uv_map = np.argsort(uv_names)
        for ii in uv_map:
            hash_update(uv_names[ii].encode(encoding='utf-8'))
            uvs_raw = seq2numpyarray(mesh.uv_layers[ii].data, 'uv', np.float32, 2)
            if (useRound):
                np.around(uvs_raw, decimals=ROUND, out=uvs_raw)
            hash_mapped(hash_update, uvs_raw, loops_map, [('x', np.float32), ('y', np.float32)])

        # vertex groups
        # not really happy with this, but I don't know a more efficient way to get all weights
        vgroup_weights = [[0.0] * len(mesh.vertices) for i in obj.vertex_groups] #empty array for all weights

        for vert in mesh.vertices:
            for weight in vert.groups:
                vgroup_weights[weight.index][vert.index] = weight.weight

        vgroup_names = [group.name for group in obj.vertex_groups]
        vgroup_map = np.argsort(vgroup_names)
        for ii in vgroup_map:
            hash_update(vgroup_names[ii].encode(encoding='utf-8'))
            weights_sorted = np.array(vgroup_weights[ii], dtype=np.float32)[verts_map]
            if (sys.byteorder != 'little'):
                weights_sorted.byteswap(True)
            hash_update(weights_sorted)

        vcolor_names = [color.name for color in mesh.vertex_colors]
        vcolor_map = np.argsort(vcolor_names)
        for ii in vcolor_map:
            hash_update(vcolor_names[ii].encode(encoding='utf-8'))
            vcolors_raw = seq2numpyarray(mesh.vertex_colors[ii].data, 'color', np.float32, 4) # vertex colors have 4 components. getting as 8 bit integer will not work
            hash_mapped(hash_update, vcolors_raw, loops_map, [('r', np.float32), ('g', np.float32), ('b', np.float32), ('a', np.float32)])

        # sharp edges
        edges_sharp_raw = seq2numpyarray(mesh.edges, 'use_edge_sharp', np.uint8, 1)
        edges_sharp_sorted = edges_sharp_raw[edges_map]

        # materials
        # materials are hashed based on original material index, names are not sorted,
        # because they are global names and depending on the content of the scene,
        # a name may not be available and has to be changed

        materials_raw = seq2numpyarray(mesh.polygons, 'material_index', np.uint32, 1)
        materials_sorted = materials_raw[polys_map] # reorder polygons


        # do hashing
        if (sys.byteorder != 'little'):
            verts_sorted.byteswap(True)
            loops_sorted.byteswap(True)
            edges_sorted.byteswap(True)
            #edges_sharp_sorted.byteswap(True) # 8 bit type
            materials_sorted.byteswap(True)

        hash_update(verts_sorted)
        hash_update(loops_sorted) # we don't hash polys directly, but since we hash the loops, that's already in there implicitly
        hash_update(edges_sorted)
        hash_update(edges_sharp_sorted)
        hash_update(materials_sorted)

    import hashlib

    md5 = hashlib.new("md5")
    md5_update = md5.update

    for obj in scene.objects:
        md5_update(matrix2str(obj.matrix_world))
        data = obj.data

        if type(data) == bpy.types.Mesh:
            #mesh_hash(md5_update, data)
            mesh_hash_sorted(md5_update, obj)
        elif type(data) == bpy.types.Curve:
            for spline in data.splines:
                md5_update(coords2str(spline.bezier_points, "co"))
                md5_update(coords2str(spline.points, "co"))

    return md5.hexdigest()


def main():
    argv = sys.argv
    print("  args:", " ".join(argv))
    argv = argv[argv.index("--") + 1:]

    def arg_extract(arg, optional=True, array=False):
        arg += "="
        if array:
            value = []
        else:
            value = None

        i = 0
        while i < len(argv):
            if argv[i].startswith(arg):
                item = argv[i][len(arg):]
                del argv[i]
                i -= 1

                if array:
                    value.append(item)
                else:
                    value = item
                    break

            i += 1

        if (not value) and (not optional):
            print("  '%s' not set" % arg)
            sys.exit(1)

        return value

    run = arg_extract("--run", optional=False)
    md5 = arg_extract("--md5", optional=False)
    md5_method = arg_extract("--md5_method", optional=False)  # 'SCENE' / 'FILE'

    # only when md5_method is 'FILE'
    md5_source = arg_extract("--md5_source", optional=True, array=True)

    # save blend file, for testing
    write_blend = arg_extract("--write-blend", optional=True)

    # ensure files are written anew
    for f in md5_source:
        if os.path.exists(f):
            os.remove(f)

    import bpy

    replace_bpy_app_version()
    if not bpy.data.filepath:
        clear_startup_blend()

    print("  Running: '%s'" % run)
    print("  MD5: '%s'!" % md5)

    try:
        result = eval(run)
    except:
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if write_blend is not None:
        print("  Writing Blend: %s" % write_blend)
        bpy.ops.wm.save_mainfile('EXEC_DEFAULT', filepath=write_blend)

    print("  Result: '%s'" % str(result))
    if not result:
        print("  Running: %s -> False" % run)
        sys.exit(1)

    if md5_method == 'SCENE':
        md5_new = blend_to_md5()
    elif md5_method == 'FILE':
        if not md5_source:
            print("  Missing --md5_source argument")
            sys.exit(1)

        for f in md5_source:
            if not os.path.exists(f):
                print("  Missing --md5_source=%r argument does not point to a file")
                sys.exit(1)

        import hashlib

        md5_instance = hashlib.new("md5")
        md5_update = md5_instance.update

        for f in md5_source:
            filehandle = open(f, "rb")
            md5_update(filehandle.read())
            filehandle.close()

        md5_new = md5_instance.hexdigest()

    else:
        print("  Invalid --md5_method=%s argument is not a valid source")
        sys.exit(1)

    if md5 != md5_new:
        print("  Running: %s\n    MD5 Recieved: %s\n    MD5 Expected: %s" % (run, md5_new, md5))
        sys.exit(1)

    print("  Success: %s" % run)


if __name__ == "__main__":
    # So a python error exits(1)
    try:
        main()
    except:
        import traceback
        traceback.print_exc()
        sys.exit(1)
