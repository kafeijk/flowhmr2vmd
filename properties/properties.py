"""Panel properties and finger-limit constraint state."""
import bpy

from ..core import finger_limits, mmd
from ..core.constants import MAPPING_ITEMS

_LIMIT_MIN = -180.0
_LIMIT_MAX = 180.0


def _active_armature(context):
    obj = context.active_object
    if obj is None or obj not in context.selected_objects:
        selected = list(context.selected_objects)
        obj = selected[0] if selected else obj
    if obj is None:
        return None
    root, armature = mmd.find_root_and_armature(obj)
    return armature if root is not None else None


def _apply(self, context, armature):
    count, missing = finger_limits.apply_finger_limits(armature, self)
    status = f'已创建 {count} 根手指限制约束。'
    if missing:
        status += f'缺少 {len(missing)} 根：' + '、'.join(missing[:4])
        if len(missing) > 4:
            status += '…'
    self.finger_limit_status = status


def _update_limit(self, context):
    if not self.finger_limits_enabled:
        return
    armature = _active_armature(context)
    if armature is None:
        self.finger_limits_enabled = False
        self.finger_limit_status = '请先选中 MMD 模型对象。'
        return
    _apply(self, context, armature)


def _update_enabled(self, context):
    armature = _active_armature(context)
    if not self.finger_limits_enabled:
        if armature is not None:
            count = finger_limits.remove_finger_limits(armature)
        else:
            count = sum(
                finger_limits.remove_finger_limits(obj)
                for obj in context.scene.objects if obj.type == 'ARMATURE')
        self.finger_limit_status = f'已删除 {count} 根手指限制约束。'
        return
    if armature is None:
        self.finger_limits_enabled = False
        self.finger_limit_status = '请先选中 MMD 模型对象。'
        return
    _apply(self, context, armature)


class RGBMOCAP_Properties(bpy.types.PropertyGroup):
    folder: bpy.props.StringProperty(
        name='文件夹',
        description='包含 final.npz 或 RGB MoCap 工程 flowhmr 缓存的文件夹',
        subtype='DIR_PATH',
    )
    mapping: bpy.props.EnumProperty(
        name='名称对照',
        description='选择目标模型的骨骼命名方式',
        items=MAPPING_ITEMS,
        default='FALLBACK',
    )

    show_finger_limits: bpy.props.BoolProperty(
        name='展开手指骨骼约束',
        description='展开后可在下方修改六组基础手指的 XYZ 最小/最大角度',
        default=False,
    )
    finger_limits_enabled: bpy.props.BoolProperty(
        name='创建手指限制旋转约束',
        description='开启后为 18 根手指创建 LIMIT_ROTATION 约束；关闭时只删除本插件添加的约束',
        default=False,
        update=_update_enabled,
    )
    finger_limit_status: bpy.props.StringProperty(
        name='手指约束状态',
        default='',
    )

    thumb0_x_min: bpy.props.FloatProperty(name='親指０ X 最小', default=-30.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb0_x_max: bpy.props.FloatProperty(name='親指０ X 最大', default=60.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb0_y_min: bpy.props.FloatProperty(name='親指０ Y 最小', default=-30.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb0_y_max: bpy.props.FloatProperty(name='親指０ Y 最大', default=60.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb0_z_min: bpy.props.FloatProperty(name='親指０ Z 最小', default=-25.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb0_z_max: bpy.props.FloatProperty(name='親指０ Z 最大', default=35.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)

    thumb1_x_min: bpy.props.FloatProperty(name='親指１ X 最小', default=-30.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb1_x_max: bpy.props.FloatProperty(name='親指１ X 最大', default=60.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb1_y_min: bpy.props.FloatProperty(name='親指１ Y 最小', default=-20.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb1_y_max: bpy.props.FloatProperty(name='親指１ Y 最大', default=70.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb1_z_min: bpy.props.FloatProperty(name='親指１ Z 最小', default=-50.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb1_z_max: bpy.props.FloatProperty(name='親指１ Z 最大', default=65.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)

    thumb2_x_min: bpy.props.FloatProperty(name='親指２ X 最小', default=-15.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb2_x_max: bpy.props.FloatProperty(name='親指２ X 最大', default=30.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb2_y_min: bpy.props.FloatProperty(name='親指２ Y 最小', default=-10.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb2_y_max: bpy.props.FloatProperty(name='親指２ Y 最大', default=30.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb2_z_min: bpy.props.FloatProperty(name='親指２ Z 最小', default=-70.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    thumb2_z_max: bpy.props.FloatProperty(name='親指２ Z 最大', default=30.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)

    index1_x_min: bpy.props.FloatProperty(name='人指１ X 最小', default=-30.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index1_x_max: bpy.props.FloatProperty(name='人指１ X 最大', default=90.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index1_y_min: bpy.props.FloatProperty(name='人指１ Y 最小', default=-12.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index1_y_max: bpy.props.FloatProperty(name='人指１ Y 最大', default=6.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index1_z_min: bpy.props.FloatProperty(name='人指１ Z 最小', default=-15.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index1_z_max: bpy.props.FloatProperty(name='人指１ Z 最大', default=30.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)

    index2_x_min: bpy.props.FloatProperty(name='人指２ X 最小', default=-25.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index2_x_max: bpy.props.FloatProperty(name='人指２ X 最大', default=140.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index2_y_min: bpy.props.FloatProperty(name='人指２ Y 最小', default=-3.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index2_y_max: bpy.props.FloatProperty(name='人指２ Y 最大', default=1.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index2_z_min: bpy.props.FloatProperty(name='人指２ Z 最小', default=-3.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index2_z_max: bpy.props.FloatProperty(name='人指２ Z 最大', default=1.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)

    index3_x_min: bpy.props.FloatProperty(name='人指３ X 最小', default=-20.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index3_x_max: bpy.props.FloatProperty(name='人指３ X 最大', default=85.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index3_y_min: bpy.props.FloatProperty(name='人指３ Y 最小', default=-1.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index3_y_max: bpy.props.FloatProperty(name='人指３ Y 最大', default=6.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index3_z_min: bpy.props.FloatProperty(name='人指３ Z 最小', default=-15.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)
    index3_z_max: bpy.props.FloatProperty(name='人指３ Z 最大', default=1.0,
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit)

    @staticmethod
    def register():
        if not hasattr(bpy.types.Scene, 'rgbmocap_properties'):
            bpy.types.Scene.rgbmocap_properties = bpy.props.PointerProperty(
                type=RGBMOCAP_Properties)

    @staticmethod
    def unregister():
        if hasattr(bpy.types.Scene, 'rgbmocap_properties'):
            del bpy.types.Scene.rgbmocap_properties


def register():
    RGBMOCAP_Properties.register()


def unregister():
    RGBMOCAP_Properties.unregister()
