"""
Created on Apr 22, 2016

"""

import functools
import operator

import numpy as np


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
