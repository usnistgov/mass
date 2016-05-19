"""
Created on Apr 22, 2016

"""

import functools
import operator

import numpy as np


class CutFieldMixin(object):
    BUILTIN_BOOLEAN_CUT_FIELDS = ['pretrigger_rms',
                                  'pretrigger_mean',
                                  'pretrigger_mean_departure_from_median',
                                  'peak_time_ms',
                                  'rise_time_ms',
                                  'postpeak_deriv',
                                  'pulse_average',
                                  'min_value',
                                  'timestamp_sec',
                                  'timestamp_diff_sec',
                                  'peak_value',
                                  'energy',
                                  'timing',
                                  "p_filt_phase",
                                  'smart_cuts']

    # Categorical cut field item format
    # [name of field, list of categories, default category]


    BUILTIN_CATEGORICAL_CUT_FIELDS = [
        ['calibration', ['in', 'out'], 'in'],
    ]

    CUT_BOOLEAN_FIELD_DESC_DTYPE = np.dtype([("name", np.bytes_, 64),
                                             ("mask", np.uint32)])

    CUT_CATEGORICAL_FIELD_DESC_DTYPE = np.dtype([("name", np.bytes_, 64),
                                                 ("mask", np.uint32)])

    CUT_CATEGORY_LIST_DTYPE = np.dtype([("field", np.bytes_, 64),
                                        ("category", np.bytes_, 64),
                                        ("code", np.uint32)])

    def cut_field_desc_init(self):
        if self.hdf5_file:
            if 'cut_num_used_bits' in self.hdf5_file.attrs:
                self.hdf5_file.attrs['cut_format_ver'] = b'1'

                # convert from version 1 to the verion 2
                cut_num_used_bits = np.uint32(self.hdf5_file.attrs["cut_num_used_bits"])
                self.hdf5_file.attrs['cut_used_bit_flags'] = np.uint32((np.uint64(1) << cut_num_used_bits) - 1)

                self.boolean_cut_desc = np.asarray(self.boolean_cut_desc,
                                                   dtype=self.CUT_BOOLEAN_FIELD_DESC_DTYPE)
                self.categorical_cut_desc = np.asarray(self.categorical_cut_desc[['name', 'mask']],
                                                       dtype=self.CUT_CATEGORICAL_FIELD_DESC_DTYPE)
                self.cut_category_list = np.asarray(list(self.cut_category_list),
                                                    dtype=self.CUT_CATEGORY_LIST_DTYPE)

                del self.hdf5_file.attrs['cut_num_used_bits']

            # here, we can assume that cut descriptions are empty or version 2
            if "cut_used_bit_flags" not in self.hdf5_file.attrs:
                self.hdf5_file.attrs['cut_used_bit_flags'] = np.uint32(0)

            if "cut_boolean_field_desc" not in self.hdf5_file.attrs:
                self.boolean_cut_desc = np.zeros(32, dtype=self.CUT_BOOLEAN_FIELD_DESC_DTYPE)
                self.register_boolean_cut_fields(*self.BUILTIN_BOOLEAN_CUT_FIELDS)

            if ("cut_categorical_field_desc" not in self.hdf5_file.attrs) and \
                    ("cut_category_list" not in self.hdf5_file.attrs):
                self.categorical_cut_desc = np.zeros(0, dtype=self.CUT_CATEGORICAL_FIELD_DESC_DTYPE)
                self.cut_category_list = np.zeros(0, dtype=self.CUT_CATEGORY_LIST_DTYPE)

                for categorical_desc in self.BUILTIN_CATEGORICAL_CUT_FIELDS:
                    self.register_categorical_cut_field(*categorical_desc)

            self.hdf5_file.attrs['cut_format_ver'] = b'2'

    @property
    def boolean_cut_desc(self):
        return self.hdf5_file.attrs["cut_boolean_field_desc"]

    @boolean_cut_desc.setter
    def boolean_cut_desc(self, value):
        self.hdf5_file.attrs["cut_boolean_field_desc"] = value

    @property
    def categorical_cut_desc(self):
        return self.hdf5_file.attrs["cut_categorical_field_desc"]

    @categorical_cut_desc.setter
    def categorical_cut_desc(self, value):
        self.hdf5_file.attrs["cut_categorical_field_desc"] = value

    @property
    def cut_category_list(self):
        return self.hdf5_file.attrs["cut_category_list"]

    @cut_category_list.setter
    def cut_category_list(self, value):
        self.hdf5_file.attrs["cut_category_list"] = value

    @property
    def cut_used_bit_flags(self):
        return self.hdf5_file.attrs["cut_used_bit_flags"]

    @cut_used_bit_flags.setter
    def cut_used_bit_flags(self, value):
        self.hdf5_file.attrs["cut_used_bit_flags"] = np.uint32(value)

    def cut_field_categories(self, field_name):
        category_list = self.cut_category_list

        return {category.decode(): code for field, category, code in category_list if field == field_name.encode()}

    @staticmethod
    def __lowest_available_cut_bit(cut_used_bit_flags):
        """Returns the index of lowest available cut bit.

        Args:
            cut_used_bit_flags (np.uint32): This number represents a status of used cut bits.
                It doesn't need to be same with the current status of cut bits.

        Return:
            np.uint32
        """
        uint32_one = np.uint32(1)

        for i in range(32):
            trial_bit_pos = np.uint32(i)
            trial_bit = uint32_one << trial_bit_pos
            if cut_used_bit_flags & trial_bit == 0:
                return trial_bit_pos

        raise ValueError("No available cut bit.")

    def register_boolean_cut_fields(self, *names):
        """Register one or more boolean cut field(s).
        If any of given boolean cut fields already exist, it silently ignore.

         Args:
             names (list[str]): name(s) of one or more cut fields(s).
        """
        boolean_fields = self.boolean_cut_desc
        cut_used_bit_flags = self.cut_used_bit_flags

        new_fields = [n.encode() for n in names if n.encode() not in boolean_fields["name"]]

        uint32_one = np.uint32(1)
        for new_field in new_fields:
            available_bit_pos = self.__lowest_available_cut_bit(cut_used_bit_flags)
            boolean_fields[available_bit_pos] = (new_field, uint32_one << available_bit_pos)
            cut_used_bit_flags |= (uint32_one << available_bit_pos)

        self.boolean_cut_desc = boolean_fields
        self.cut_used_bit_flags = cut_used_bit_flags

    def unregister_boolean_cut_fields(self, *names):
        """Unregister one or more boolean cut fields.

        Args:
            names (list[str]): one or more name(s) of boolean cut fields.

        Raise:
            KeyError: when any of cut fields don't exist.
        """
        boolean_fields = self.boolean_cut_desc

        enc_names = [name.encode() for name in names]

        for name in enc_names:
            if not name or name not in boolean_fields['name']:
                raise KeyError("{0:s} is not a registered boolean field.".format(name.decode()))

        clear_mask = np.uint32(0)

        for i in range(32):
            if boolean_fields[i][0] in enc_names:
                clear_mask |= boolean_fields[i][1]
                boolean_fields[i] = (b'', 0)

        self.boolean_cut_desc = boolean_fields
        self.cut_used_bit_flags &= ~clear_mask

    def register_categorical_cut_field(self, name, categories, default="uncategorized"):
        """Register one categorical cut field.

        Args:
            name (str): the name of a new categorical cut field.
            categories (list[str]): the list of the names of categories of the cut field.
                "uncategorized" category will be added if it doesn't have already.
            default (str): the name of default category.
        """
        categorical_fields = self.categorical_cut_desc
        cut_used_bit_flags = self.cut_used_bit_flags

        if name.encode() in categorical_fields["name"]:
            return

        # categories might be an immutable tuple.
        # And it converts categories into str(s).
        category_list = [str(category) for category in categories]

        default = str(default)

        # if the default category is already included, it's temporarily removed from the category_list
        # and insert into at the head of the category_list.
        if default in category_list:
            category_list.remove(default)
        category_list.insert(0, default)

        num_bits = 1
        while (1 << num_bits) < len(category_list):
            num_bits += 1

        individual_bit_masks = []
        bit_mask = np.uint32(0)
        lowest_bit_pos = np.uint32(31)
        uint32_one = np.uint32(1)

        for _ in range(num_bits):
            bit_pos = self.__lowest_available_cut_bit(cut_used_bit_flags | bit_mask)
            if bit_pos < lowest_bit_pos:
                lowest_bit_pos = bit_pos
            bit_mask |= (uint32_one << bit_pos)
            individual_bit_masks.insert(0, uint32_one << bit_pos)

        # Updates the 'cut_category_list' attribute
        new_list = []
        for i, category in enumerate(category_list):
            digits = map(np.uint32, "{0:032b}".format(i)[-num_bits:])
            code = np.sum([a * b for a, b in zip(individual_bit_masks, digits)])
            new_list.append((name.encode(), category.encode(), code >> lowest_bit_pos))

        new_list = np.array(new_list, dtype=self.CUT_CATEGORY_LIST_DTYPE)
        self.cut_category_list = np.hstack([self.cut_category_list, new_list])

        # Needs to update the 'cut_categorical_field_desc' attribute.
        field_desc_item = np.array([(name.encode(), bit_mask)],
                                   dtype=self.CUT_CATEGORICAL_FIELD_DESC_DTYPE)
        self.categorical_cut_desc = np.hstack([categorical_fields, field_desc_item])
        self.cut_used_bit_flags |= bit_mask

    def unregister_categorical_cut_field(self, name):
        """Unregister one categorical cut field

        Args:
            name (str): the name of a categorical cut field to be unregistered.
        """
        categorical_fields = self.categorical_cut_desc
        category_list = self.cut_category_list

        if not np.any(categorical_fields['name'] == name.encode()):
            raise ValueError("{0:s} field is not a registered categorical field.".format(name))

        new_categorical_fields = categorical_fields[categorical_fields['name'] != name.encode()]
        new_category_list = category_list[category_list['field'] != name.encode()]
        clear_mask = categorical_fields['mask'][categorical_fields['name'] == name.encode()][0]

        self.categorical_cut_desc = new_categorical_fields
        self.cut_category_list = new_category_list
        self.cut_used_bit_flags &= ~clear_mask


class Cuts(object):
    """
    Object to hold a 32-bit cut mask for each triggered record.
    """

    def __init__(self, n, tes_group, hdf5_group=None):
        """
        Create an object to hold n masks of 32 bits each
        """
        self.tes_group = tes_group
        self.hdf5_group = hdf5_group
        if hdf5_group is None:
            self._mask = np.zeros(n, dtype=np.uint32)
        else:
            try:
                self._mask = hdf5_group.require_dataset('mask', shape=(n,), dtype=np.uint32)
            except TypeError:
                temp = hdf5_group.require_dataset('mask', shape=(n,), dtype=np.int32)[...]
                del hdf5_group['mask']
                self._mask = hdf5_group.require_dataset('mask', shape=(n,), dtype=np.uint32)
                self._mask[...] = np.asarray(temp, dtype=np.uint32)

    def cut(self, cut_num, mask):
        """
        Set the mask of a single field. It could be a boolean or categorical field.
        """
        assert(mask.size == self._mask.size)

        boolean_field = self.tes_group.boolean_cut_desc
        categorical_field = self.tes_group.categorical_cut_desc

        if isinstance(cut_num, int) or isinstance(cut_num, np.uint) or isinstance(cut_num, np.int):
            cut_num = int(cut_num)
            if (cut_num < 0) or (cut_num > 31):
                raise ValueError(str(cut_num) + " is out of range.")
            if boolean_field[cut_num]['name'] == ''.encode():
                raise ValueError(str(cut_num) + " is not a registered boolean cut.")
            _, bit_mask = boolean_field[cut_num]
            self._mask[mask] |= bit_mask
        elif isinstance(cut_num, bytes) or isinstance(cut_num, str):
            # This condition will work because we don't expect Python 2.7 users to pass an unicode cut_num.
            boolean_g = (boolean_field["name"] == cut_num.encode())
            if np.any(boolean_g):
                _, bit_mask = boolean_field[boolean_g][0]
                self._mask[mask] |= bit_mask
            else:
                categorical_g = (categorical_field["name"] == cut_num.encode())
                if np.any(categorical_g):
                    _, bit_mask = categorical_field[categorical_g][0]

                    for i in range(32):
                        bit_pos = np.uint32(i)
                        if (bit_mask >> bit_pos) & np.uint32(1):
                            break

                    temp = self._mask[...] & ~bit_mask
                    category_values = np.asarray(mask, dtype=np.uint32) << bit_pos
                    self._mask[...] = temp | (category_values & bit_mask)
                else:
                    raise ValueError(cut_num + " field is not found.")

    def cut_categorical(self, field, booldict):
        """
        Args:
            field: string name of category
            booldict: dictionary with keys are category of the field and
                entries are bool vectors of length equal to make indicating belongingness
        """
        category_names = self.tes_group.cut_field_categories(field)
        labels = np.zeros(len(self._mask), dtype=np.uint32)
        for (category, catbool) in booldict.items():
            labels[catbool] = category_names[category]
        for (category, catbool) in booldict.items():
            if not all(labels[booldict[category]] == category_names[category]):
                raise ValueError("bools passed for %s conflict with some other" % category)
        self.cut(field, labels)

    def select_category(self, **kwargs):
        """
        Select pulses belongs to all of specified categories.

        Returns:
            A numpy array of booleans.
        """
        category_field_bit_mask = np.uint32(0)
        category_field_target_bits = np.uint32(0)

        categorical_fields = self.tes_group.categorical_cut_desc
        category_list = self.tes_group.cut_category_list

        for name, category_label in kwargs.items():
            categorical_g = (categorical_fields["name"] == name.encode())

            if not np.any(categorical_g):
                raise ValueError(name + " categorical field is not found.")

            category_g = (category_list["field"] == name.encode()) &\
                         (category_list["category"] == category_label.encode())

            if not np.any(category_g):
                raise ValueError(category_label + " category is not found.")

            _, bit_mask = categorical_fields[categorical_g][0]
            _, _, code = category_list[category_g][0]

            for i in range(32):
                bit_pos = np.uint32(i)
                if (bit_mask >> bit_pos) & np.uint32(1):
                    break

            category_field_bit_mask |= bit_mask
            category_field_target_bits |= code << bit_pos

        return (self._mask[...] & category_field_bit_mask) == category_field_target_bits

    def category_codes(self, name):
        """
        Returns the category codes of a single categorical cut field.

        Args:
            name : string
                the name of a categorical cut field.

        Returns:
            numpy array of uint32 :
                category codes of a categorical cut field 'name'.

        Raises:
            KeyError
                when a name is not a registered categorical cut field.
        """
        categorical_field = self.tes_group.categorical_cut_desc
        categorical_field_g = categorical_field["name"] == name.encode()

        if np.any(categorical_field_g):
            _, bit_mask = categorical_field[categorical_field_g][0]

            for i in range(32):
                bit_pos = np.uint32(i)
                if (bit_mask >> bit_pos) & np.uint32(1):
                    break
        else:
            raise KeyError(name + " is not found.")

        return (self._mask[...] & bit_mask) >> bit_pos

    def cut_mask(self, *fields):
        """
         Retrieves masks of multiple cut fields. They could be boolean or categorical.

         Args:
             fields: cut field name or names.
        """
        boolean_field = self.tes_group.boolean_cut_desc
        categorical_field = self.tes_group.categorical_cut_desc

        if fields:
            boolean_field_names = [str(name.decode()) for name, _ in boolean_field if name.decode() in fields]
            categorical_field_names = [str(name.decode()) for name, _ in categorical_field if name.decode() in fields]

            not_found = set(fields) - (set(boolean_field_names).union(set(categorical_field_names)))
            if not_found:
                raise ValueError(",".join(not_found) + " are not found.")
        else:
            boolean_field_names = [str(name.decode()) for name, _ in boolean_field if name]
            categorical_field_names = [str(name.decode()) for name, _, in categorical_field]

        mask_dtype = np.dtype([(name, np.bool) for name in boolean_field_names] +
                              [(name, np.uint32) for name in categorical_field_names])

        cut_mask = np.zeros(self._mask.shape[0], dtype=mask_dtype)

        for name in boolean_field_names:
            cut_mask[:][name] = self.good(name)

        for name in categorical_field_names:
            cut_mask[:][name] = self.category_codes(name)

        return cut_mask

    def clear_cut(self, *args):
        """
        Clear one or more boolean fields.
        If no name is given, it will clear all boolean fields.
        """
        bit_mask = self._boolean_fields_bit_mask(args)

        self._mask[:] &= ~bit_mask

    def _boolean_fields_bit_mask(self, names):
        """
        Calculate the bit mask for any combination of boolean cut fields.
        """
        boolean_fields = self.tes_group.boolean_cut_desc

        if names:
            all_field_names = set([name.decode() for name, mask in boolean_fields if name])

            not_found_fields = set(names) - all_field_names

            if not_found_fields:
                raise ValueError(", ".join(not_found_fields) + " not found.")

            bit_masks = [mask for name, mask in boolean_fields if name.decode() in names]
        else:
            bit_masks = [mask for name, mask in boolean_fields if name]

        bit_mask = functools.reduce(operator.or_, bit_masks, np.uint32(0))

        return bit_mask

    def good(self, *args, **kwargs):
        """
        Select pulses which are good for all of specified boolean cut fields.
        If any categorical cut fields are given, only pulses in the combination of categories are considered.
        """
        bit_mask = self._boolean_fields_bit_mask(args)
        g = ((self._mask[...] & bit_mask) == 0)

        if kwargs:
            return g & self.select_category(**kwargs)

        return g

    def bad(self, *args, **kwargs):
        """
        Select pulses which are bad for at least one of specified boolean cut fields.
        If any categorical cut fields are given, only pulses in the combination of categories are considered.
        """
        bit_mask = self._boolean_fields_bit_mask(args)
        g = (self._mask[...] & bit_mask != 0)

        if kwargs:
            return g & self.select_category(**kwargs)

        return g

    def __repr__(self):
        return "Cuts(%d)" % len(self._mask)

    def __str__(self):
        return "Cuts(%d) with %d cut and %d uncut" % (len(self._mask), self.bad().sum(), self.good().sum())

    def copy(self):
        """
        I don't see the point of this shallow copy.
        """
        c = Cuts(len(self._mask), tes_group=self.tes_group, hdf5_group=self.hdf5_group)
        return c
