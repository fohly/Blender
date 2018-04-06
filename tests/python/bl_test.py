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

    def floatseq2str(seq):
        return "".join([str(round(val, ROUND)) for val in seq]).encode('ASCII')

    def matrix2str(matrix):
        return floatseq2str(axis for vector in matrix for axis in vector)

    def coords2str(seq, attr):
        return floatseq2str(axis for vertex in seq for axis in getattr(vertex, attr))

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
        
    def time_help(str, timer):
        t = time.time()
        print(str, t-timer[0])
        timer[0] = t

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
            hash_sequence(hash_update, vcolor.data, 'vertices', np.float32, 4) # vertex colors have 4 components. getting as 8 bit integer will not work

        #uvs
        for uv in mesh.uv_layers:
            hash_sequence(hash_update, uv.data, 'uv', np.float32, 2)

        #vertex groups
        #TODO

    #rotates each loop in the list, so the vertex with the lowest index is first
    #then sorts the polys based on a hash based on its vertex indices
    #returns a tuple containting:
    # - an indexing from new to old loop index
    # - an indexing from new to old polygon index
    def rotate_sort_loops(loops, polys, polys_len):
        loops = loops.tolist() # we don't use numpy stuff in this function, and lists provide faster indexing
        polys = polys.tolist()
        polys_len = polys_len.tolist()
        
        # rotate verts inside poly and compute hash
        loops_index = [0] * len(loops)
        loops_hash = [0] * len(polys) # hopefully unique value per poly
        for ii in range(len(polys)):
            start_idx = polys[ii]
            length = polys_len[ii]
            min_vertex = loops[start_idx]
            min_idx = 0
            for jj in range(1, length):
                if loops[jj + start_idx] < min_vertex:
                    min_vertex = loops[jj + start_idx]
                    min_idx = jj
            loop_hash = 0
            for jj in range(length):
                idx = start_idx + ((jj + min_idx) % length)
                loops_index[jj + start_idx] = idx
                loop_hash += loops[idx] * (jj+1) * 42424243 # multiply by some random prime number, to get a nice hash
            loops_hash[ii] = loop_hash
        
        loop_reorder = np.argsort(loops_hash, kind='mergesort') # mergesort, even though we should not get collisions
        
        # reorder the polygons based on hash
        
        loops_index_2 = [0] * len(loops)
        next_start = 0
        for ii in range(len(polys)):
            poly_idx = loop_reorder[ii]
            loop_start = polys[poly_idx]
            length = polys_len[poly_idx]
            for jj in range(length):
                loops_index_2[next_start + jj] = loops_index[loop_start + jj]
            next_start += length
        
        return (loops_index_2, loop_reorder)
    
    #hashes the mesh data
    #first sorts the data to be deterministic
    def mesh_hash_sorted(hash_update, mesh):
        # prepare vertex data

        timer = [time.time()]
        
        verts_raw = seq2numpyarray(mesh.vertices, 'co', np.float32, 3)
        time_help("verts_raw", timer)
        verts = verts_raw.view(dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
        time_help("verts", timer)
        verts_index = np.argsort(verts, kind='mergesort', order=['x', 'y', 'z']) # mergesort, because it is stable (quicksort is based on random numbers usually)
        #verts_index maps from new index to old index. so verts_index[0] is the index of the "lowest" vertex
        time_help("verts_index", timer)
        verts_remapping = np.argsort(verts_index) # allows finding the new index of a vertex based on the old one
        time_help("verts_remapping", timer)
        verts_sorted = verts[verts_index]
        time_help("verts_sorted", timer)
        
        #print(verts)
        #print(verts_sorted)
        #print(verts_index)
        
        polys = seq2numpyarray(mesh.polygons, 'loop_start', np.uint32, 1)
        polys_len = seq2numpyarray(mesh.polygons, 'loop_total', np.uint32, 1)
        
        

        # prepare loop data
        loops = seq2numpyarray(mesh.loops, 'vertex_index', np.uint32, 1) # vertex indices per loop
        time_help("loops", timer)
        
        #print(loops)
        
        loops2 = verts_remapping[loops] # remap vertex indices of loops
        time_help("loops2", timer)
        
        #print(loops2)
        
        (loops_index, poly_index) = rotate_sort_loops(loops2, polys, polys_len)
        loops_sorted = loops2[loops_index]
        time_help("loops_sorted", timer)
        #print(loops_sorted)

        # do hashing
        if (sys.byteorder != 'little'):
            verts_sorted.byteswap(True)
            loops_sorted.byteswap(True)

        hash_update(verts_sorted)
        hash_update(loops_sorted)
        # we don't hash polys directly, but since we hash the loops, that's already in there implicitly
        time_help("hash_update", timer)


    import hashlib

    md5 = hashlib.new("md5")
    md5_update = md5.update

    for obj in scene.objects:
        md5_update(matrix2str(obj.matrix_world))
        data = obj.data

        if type(data) == bpy.types.Mesh:
            #mesh_hash(md5_update, data)
            mesh_hash_sorted(md5_update, data)
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
