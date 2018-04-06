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
    scene = bpy.context.scene
    ROUND = 4

    def floatseq2str(seq):
        return "".join([str(round(val, ROUND)) for val in seq]).encode('ASCII')

    def matrix2str(matrix):
        return floatseq2str(axis for vector in matrix for axis in vector)

    def coords2str(seq, attr):
        return floatseq2str(axis for vertex in seq for axis in getattr(vertex, attr))

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
            hash_sequence(hash_update, vcolor.data, 'vertices', np.float32, 4) # vertex colors have 4 components. getting as 8 bit integer will not work

        #uvs
        for uv in mesh.uv_layers:
            hash_sequence(hash_update, uv.data, 'uv', np.float32, 2)

        #vertex groups
        #TODO


    import hashlib

    md5 = hashlib.new("md5")
    md5_update = md5.update

    for obj in scene.objects:
        md5_update(matrix2str(obj.matrix_world))
        data = obj.data

        if type(data) == bpy.types.Mesh:
            mesh_hash(md5_update, data)
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
