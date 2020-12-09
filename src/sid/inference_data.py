"""Data structure for using zarr groups with xarray."""
import numpy as np
import functools
from pathlib import Path
import uuid
import warnings
from copy import deepcopy
from html import escape
import zarr
import xarray as xr
import re


SUPPORTED_GROUPS = [
    "time_series",
    "last_states",
    "sample_stats",
]


def _extend_xr_method(func):
    """Make wrapper to extend methods from xr.Dataset to InferenceData Class."""
    # pydocstyle requires a non empty line

    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        _filter = kwargs.pop("filter_groups", None)
        _groups = kwargs.pop("groups", None)
        _inplace = kwargs.pop("inplace", False)

        out = self if _inplace else deepcopy(self)

        groups = self._group_names(_groups, _filter)  # pylint: disable=protected-access
        for group in groups:
            xr_data = getattr(out, group)
            xr_data = func(xr_data, *args, **kwargs)  # pylint: disable=not-callable
            setattr(out, group, xr_data)

        return None if _inplace else out


class InferenceData:
    """Container for inference data storage using xarray.

    For a detailed introduction to ``InferenceData`` objects and their usage, see
    :ref:`xarray_for_arviz`. This page provides help and documentation on
    ``InferenceData`` methods and their low level implementation.

    """

    def __init__(self, **kwargs):
        """Initialize InferenceData object from keyword xarray datasets.

        Parameters
        ----------
        kwargs :
            Keyword arguments of xarray datasets

        """
        self._groups = []
        key_list = [key for key in SUPPORTED_GROUPS if key in kwargs]
        for key in kwargs:
            if key not in SUPPORTED_GROUPS:
                key_list.append(key)
                warnings.warn(
                    f"{key} group is not defined in the InferenceData scheme",
                    UserWarning,
                )
        for key in key_list:
            dataset = kwargs[key]
            if dataset is None:
                continue
            elif not isinstance(dataset, xr.Dataset):
                raise ValueError(
                    "Arguments to InferenceData must be xarray Datasets "
                    f"(argument '{key}' was type '{type(dataset)}')"
                )
            if dataset:
                setattr(self, key, dataset)
                self._groups.append(key)

    def __repr__(self):
        """Make string representation of InferenceData object."""
        msg = "Inference data with groups:\n\t> {options}".format(
            options="\n\t> ".join(self._groups)
        )
        return msg

    def _repr_html_(self):
        """Make html representation of InferenceData object."""
        try:
            from xarray.core.options import OPTIONS

            display_style = OPTIONS["display_style"]
            if display_style == "text":
                html_repr = f"<pre>{escape(repr(self))}</pre>"
            else:
                elements = "".join(
                    [
                        HtmlTemplate.element_template.format(
                            group_id=group + str(uuid.uuid4()),
                            group=group,
                            xr_data=getattr(  # pylint: disable=protected-access
                                self, group
                            )._repr_html_(),
                        )
                        for group in self.groups
                    ]
                )
                formatted_html_template = (  # pylint: disable=possibly-unused-variable
                    HtmlTemplate.html_template.format(elements)
                )
                css_template = HtmlTemplate.css_template
                html_repr = "%(formatted_html_template)s%(css_template)s" % locals()
        except:  # noqa: E722
            html_repr = f"<pre>{escape(repr(self))}</pre>"
        return html_repr

    def __delattr__(self, group):
        """Delete a group from the InferenceData object."""
        self._groups.remove(group)
        object.__delattr__(self, group)

    def __iter__(self):
        """Iterate over groups in InferenceData object."""
        for group in self.groups:
            yield group

    def __getitem__(self, key):
        """Get item by key."""
        if key not in self.groups:
            raise KeyError(key)
        return getattr(self, key)

    def groups(self):
        """Return all groups present in InferenceData object."""
        return self.groups

    def values(self):
        """Xarray Datasets present in InferenceData object."""
        for group in self.groups:
            yield getattr(self, group)

    def items(self):
        """Yield groups and corresponding datasets present in InferenceData object."""
        for group in self.groups:
            yield (group, getattr(self, group))

    @staticmethod
    def from_zarr(filename):
        """Initialize object from a zarr file.

        Expects that the file will have groups, each of which can be loaded by xarray.
        By default, the datasets of the InferenceData object will be lazily loaded
        instead of being loaded into memory. This behaviour is regulated by the value of
        ``az.rcParams["data.load"]``.

        Parameters
        ----------
        filename : str
            location of zarr file

        Returns
        -------
        InferenceData object

        """
        if isinstance(filename, Path):
            filename = filename.as_posix()

        groups = {}
        with zarr.open(filename, mode="r") as data:
            data_groups = list(data.group_keys())

        for group in data_groups:
            with xr.open_zarr(filename, group=group) as data:
                groups[group] = data
        return InferenceData(**groups)

    def to_zarr(self, filename, groups=None, **kwargs):
        """Write InferenceData to file using zarr.
        Parameters
        ----------
        filename : str
            Location to write to
        groups : list, optional
            Write only these groups to zarr file.
        Returns
        -------
        str
            Location of zarr file
        """
        mode = "w"  # overwrite first, then append
        if self.groups:  # check's whether a group is present or not.
            if groups is None:
                groups = self.groups
            else:
                groups = [group for group in self.groups if group in groups]

            for group in groups:
                data = getattr(self, group)
                data.to_zarr(filename, mode=mode, group=group, **kwargs)
                data.close()
                mode = "a"
        else:
            raise ValueError("There is no data to store.")

        return filename

    def to_dataframe(
        self,
        groups=None,
        filter_groups=None,
        include_coords=True,
        include_index=True,
        index_origin=None,
    ):
        """Convert InferenceData to a pandas DataFrame following xarray naming
        conventions.

        This returns dataframe in a "wide" -format, where each item in ndimensional
        array is unpacked. To access "tidy" -format, use xarray functionality found for
        each dataset.

        In case of a multiple groups, function adds a group identification to the var
        name.

        Data groups ("observed_data", "constant_data", "predictions_constant_data") are
        skipped implicitly.

        Raises TypeError if no valid groups are found.

        Parameters
        ----------
        groups: str or list of str, optional
            Groups where the transformation is to be applied. Can either be group names
            or metagroup names.
        filter_groups: {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup
            names. If "regex", interpret groups as regular expressions on the real group
            or metagroup names. A la `pandas.filter`.
        include_coords: bool
            Add coordinate values to column name (tuple).
        include_index: bool
            Add index information for multidimensional arrays.
        index_origin: {0, 1}, optional
            Starting index  for multidimensional objects. 0- or 1-based.
            Defaults to rcParams["data.index_origin"].

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing all selected groups of InferenceData object.
        """
        # pylint: disable=too-many-nested-blocks
        if not include_coords and not include_index:
            raise TypeError("Both include_coords and include_index can not be False.")
        if index_origin not in [0, 1]:
            raise TypeError("index_origin must be 0 or 1, saw {}".format(index_origin))

        group_names = list(
            filter(lambda x: "data" not in x, self._group_names(groups, filter_groups))
        )

        if not group_names:
            raise TypeError("No valid groups found: {}".format(groups))

        dfs = {}
        for group in group_names:
            dataset = self[group]
            df = None
            coords_to_idx = {
                name: dict(
                    map(reversed, enumerate(dataset.coords[name].values, index_origin))
                )
                for name in list(
                    filter(lambda x: x not in ("index", "date"), dataset.coords)
                )
            }
            for data_array in dataset.values():
                dataframe = data_array.to_dataframe()
                if list(filter(lambda x: x not in ("index", "date"), data_array.dims)):
                    levels = [
                        idx
                        for idx, dim in enumerate(data_array.dims)
                        if dim not in ("index", "date")
                    ]
                    dataframe = dataframe.unstack(level=levels)
                    tuple_columns = []
                    for name, *coords in dataframe.columns:
                        if include_index:
                            idxs = []
                            for coordname, coorditem in zip(
                                dataframe.columns.names[1:], coords
                            ):
                                idxs.append(coords_to_idx[coordname][coorditem])
                            if include_coords:
                                tuple_columns.append(
                                    (
                                        "{}[{}]".format(name, ",".join(map(str, idxs))),
                                        *coords,
                                    )
                                )
                            else:
                                tuple_columns.append(
                                    "{}[{}]".format(name, ",".join(map(str, idxs)))
                                )
                        else:
                            tuple_columns.append((name, *coords))

                    dataframe.columns = tuple_columns
                    dataframe.sort_index(axis=1, inplace=True)
                if df is None:
                    df = dataframe
                    continue
                df = df.join(dataframe, how="outer")
            df = df.reset_index()
            dfs[group] = df
        if len(dfs) > 1:
            for group, df in dfs.items():
                df.columns = [
                    col
                    if col in ("index", "date")
                    else (group, col)
                    if not isinstance(col, tuple)
                    else (group, *col)
                    for col in df.columns
                ]
            dfs, *dfs_tail = list(dfs.values())
            for df in dfs_tail:
                dfs = dfs.merge(df, how="outer", copy=False)
        else:
            (dfs,) = dfs.values()
        return dfs

    def sel(
        self,
        groups=None,
        filter_groups=None,
        inplace=False,
        chain_prior=None,
        **kwargs,
    ):
        """Perform an xarray selection on all groups.

        Loops groups to perform Dataset.sel(key=item)
        for every kwarg if key is a dimension of the dataset.
        One example could be performing a burn in cut on the InferenceData object
        or discarding a chain. The selection is performed on all relevant groups (like
        posterior, prior, sample stats) while non relevant groups like observed data are
        omitted. See :meth:`xarray.Dataset.sel <xarray:xarray.Dataset.sel>`

        Parameters
        ----------
        groups: str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups: {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup
            names. If "regex", interpret groups as regular expressions on the real group
            or metagroup names. A la `pandas.filter`.
        inplace: bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.
        chain_prior: bool, optional, deprecated
            If ``False``, do not select prior related groups using ``chain`` dim.
            Otherwise, use selection on ``chain`` if present. Default=False
        **kwargs: mapping
            It must be accepted by Dataset.sel().

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform selection in-place and return `None`

        """
        if chain_prior is not None:
            warnings.warn(
                "chain_prior has been deprecated. Use groups argument and "
                "rcParams['data.metagroups'] instead.",
                DeprecationWarning,
            )
        else:
            chain_prior = False
        groups = self._group_names(groups, filter_groups)

        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            valid_keys = set(kwargs.keys()).intersection(dataset.dims)
            if not chain_prior and "prior" in group:
                valid_keys -= {"chain"}
            dataset = dataset.sel(**{key: kwargs[key] for key in valid_keys})
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def isel(
        self,
        groups=None,
        filter_groups=None,
        inplace=False,
        **kwargs,
    ):
        """Perform an xarray selection on all groups.

        Loops groups to perform Dataset.isel(key=item)
        for every kwarg if key is a dimension of the dataset.
        One example could be performing a burn in cut on the InferenceData object
        or discarding a chain. The selection is performed on all relevant groups (like
        posterior, prior, sample stats) while non relevant groups like observed data are
        omitted. See :meth:`xarray:xarray.Dataset.isel`

        Parameters
        ----------
        groups: str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups: {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup
            names. If "regex", interpret groups as regular expressions on the real group
            or metagroup names. A la `pandas.filter`.
        inplace: bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.
        **kwargs: mapping
            It must be accepted by :meth:`xarray:xarray.Dataset.isel`.

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform selection in-place and return `None`

        """
        groups = self._group_names(groups, filter_groups)

        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            valid_keys = set(kwargs.keys()).intersection(dataset.dims)
            dataset = dataset.isel(**{key: kwargs[key] for key in valid_keys})
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def stack(
        self,
        dimensions=None,
        groups=None,
        filter_groups=None,
        inplace=False,
        **kwargs,
    ):
        """Perform an xarray stacking on all groups.

        Stack any number of existing dimensions into a single new dimension.
        Loops groups to perform Dataset.stack(key=value)
        for every kwarg if value is a dimension of the dataset.
        The selection is performed on all relevant groups (like
        posterior, prior, sample stats) while non relevant groups like observed data are
        omitted. See :meth:`xarray:xarray.Dataset.stack`

        Parameters
        ----------
        dimensions: dict
            Names of new dimensions, and the existing dimensions that they replace.
        groups: str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups: {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup
            names. If "regex", interpret groups as regular expressions on the real group
            or metagroup names. A la `pandas.filter`.
        inplace: bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.
        **kwargs: mapping
            It must be accepted by :meth:`xarray:xarray.Dataset.stack`.

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform selection in-place and return `None`

        """
        groups = self._group_names(groups, filter_groups)

        dimensions = {} if dimensions is None else dimensions
        dimensions.update(kwargs)
        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            kwarg_dict = {}
            for key, value in dimensions.items():
                if not set(value).difference(dataset.dims):
                    kwarg_dict[key] = value
            dataset = dataset.stack(**kwarg_dict)
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def unstack(self, dim=None, groups=None, filter_groups=None, inplace=False):
        """Perform an xarray unstacking on all groups.

        Unstack existing dimensions corresponding to MultiIndexes into multiple new
        dimensions. Loops groups to perform Dataset.unstack(key=value). The selection is
        performed on all relevant groups (like posterior, prior, sample stats) while non
        relevant groups like observed data are omitted. See
        :meth:`xarray:xarray.Dataset.unstack`

        Parameters
        ----------
        dim: Hashable or iterable of Hashable, optional
            Dimension(s) over which to unstack. By default unstacks all MultiIndexes.
        groups: str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups: {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup
            names. If "regex", interpret groups as regular expressions on the real group
            or metagroup names. A la `pandas.filter`.
        inplace: bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform selection in place and return `None`

        """
        groups = self._group_names(groups, filter_groups)
        if isinstance(dim, str):
            dim = [dim]

        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            valid_dims = set(dim).intersection(dataset.dims) if dim is not None else dim
            dataset = dataset.unstack(dim=valid_dims)
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def rename(self, name_dict=None, groups=None, filter_groups=None, inplace=False):
        """Perform xarray renaming of variable and dimensions on all groups.

        Loops groups to perform Dataset.rename(name_dict)
        for every key in name_dict if key is a dimension/data_vars of the dataset.
        The renaming is performed on all relevant groups (like
        posterior, prior, sample stats) while non relevant groups like observed data are
        omitted. See :meth:`xarray:xarray.Dataset.rename`

        Parameters
        ----------
        name_dict: dict
            Dictionary whose keys are current variable or dimension names
            and whose values are the desired names.
        groups: str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups: {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup
            names. If "regex", interpret groups as regular expressions on the real group
            or metagroup names. A la `pandas.filter`.
        inplace: bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.


        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform renaming in-place and return `None`

        """
        groups = self._group_names(groups, filter_groups)
        if "chain" in name_dict.keys() or "draw" in name_dict.keys():
            raise KeyError("'chain' or 'draw' dimensions can't be renamed")
        out = self if inplace else deepcopy(self)

        for group in groups:
            dataset = getattr(self, group)
            expected_keys = list(dataset.data_vars) + list(dataset.dims)
            valid_keys = set(name_dict.keys()).intersection(expected_keys)
            dataset = dataset.rename({key: name_dict[key] for key in valid_keys})
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def rename_vars(
        self, name_dict=None, groups=None, filter_groups=None, inplace=False
    ):
        """Perform xarray renaming of variable or coordinate names on all groups.

        Loops groups to perform Dataset.rename_vars(name_dict) for every key in
        name_dict if key is a variable or coordinate names of the dataset. The renaming
        is performed on all relevant groups (like posterior, prior, sample stats) while
        non relevant groups like observed data are omitted. See
        :meth:`xarray:xarray.Dataset.rename_vars`

        Parameters
        ----------
        name_dict: dict
            Dictionary whose keys are current variable or coordinate names
            and whose values are the desired names.
        groups: str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups: {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup
            names. If "regex", interpret groups as regular expressions on the real group
            or metagroup names. A la `pandas.filter`.
        inplace: bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.


        Returns
        -------
        InferenceData
            A new InferenceData object with renamed variables including coordinates by
            default. When `inplace==True` perform renaming in-place and return `None`

        """
        groups = self._group_names(groups, filter_groups)

        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            valid_keys = set(name_dict.keys()).intersection(dataset.data_vars)
            dataset = dataset.rename_vars({key: name_dict[key] for key in valid_keys})
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def rename_dims(
        self, name_dict=None, groups=None, filter_groups=None, inplace=False
    ):
        """Perform xarray renaming of dimensions on all groups.

        Loops groups to perform Dataset.rename_dims(name_dict)
        for every key in name_dict if key is a dimension of the dataset.
        The renaming is performed on all relevant groups (like
        posterior, prior, sample stats) while non relevant groups like observed data are
        omitted. See :meth:`xarray:xarray.Dataset.rename_dims`

        Parameters
        ----------
        name_dict: dict
            Dictionary whose keys are current dimension names and whose values are the
            desired names.
        groups: str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups: {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup
            names. If "regex", interpret groups as regular expressions on the real group
            or metagroup names. A la `pandas.filter`.
        inplace: bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.


        Returns
        -------
        InferenceData
            A new InferenceData object with renamed dimension by default.
            When `inplace==True` perform renaming in-place and return `None`

        """
        groups = self._group_names(groups, filter_groups)
        if "chain" in name_dict.keys() or "draw" in name_dict.keys():
            raise KeyError("'chain' or 'draw' dimensions can't be renamed")

        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            valid_keys = set(name_dict.keys()).intersection(dataset.dims)
            dataset = dataset.rename_dims({key: name_dict[key] for key in valid_keys})
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def add_groups(self, group_dict=None, coords=None, dims=None, **kwargs):
        """Add new groups to InferenceData object.

        Parameters
        ----------
        group_dict: dict of {str : dict or xarray.Dataset}, optional
            Groups to be added
        coords : dict[str] -> ndarray
            Coordinates for the dataset
        dims : dict[str] -> list[str]
            Dimensions of each variable. The keys are variable names, values are lists
            of coordinates.
        **kwargs: mapping
            The keyword arguments form of group_dict. One of group_dict or kwargs must
            be provided.

        See Also
        --------
        extend : Extend InferenceData with groups from another InferenceData.
        concat : Concatenate InferenceData objects.
        """
        group_dict = either_dict_or_kwargs(group_dict, kwargs, "add_groups")
        if not group_dict:
            raise ValueError("One of group_dict or kwargs must be provided.")
        repeated_groups = [
            group for group in group_dict.keys() if group in self._groups
        ]
        if repeated_groups:
            raise ValueError("{} group(s) already exists.".format(repeated_groups))
        for group, dataset in group_dict.items():
            if group not in SUPPORTED_GROUPS:
                warnings.warn(
                    "The group {} is not defined in the InferenceData scheme".format(
                        group
                    ),
                    UserWarning,
                )
            if dataset is None:
                continue
            elif isinstance(dataset, dict):
                if (
                    group
                    in ("observed_data", "constant_data", "predictions_constant_data")
                    or group not in SUPPORTED_GROUPS
                ):
                    warnings.warn(
                        "the default dims 'chain' and 'draw' will be added "
                        "automatically",
                        UserWarning,
                    )
                dataset = dict_to_dataset(dataset, coords=coords, dims=dims)
            elif isinstance(dataset, xr.DataArray):
                if dataset.name is None:
                    dataset.name = "x"
                dataset = dataset.to_dataset()
            elif not isinstance(dataset, xr.Dataset):
                raise ValueError(
                    "Arguments to add_groups() must be xr.Dataset, xr.Dataarray or "
                    "dicts (argument '{}' was type '{}')".format(group, type(dataset))
                )
            if dataset:
                setattr(self, group, dataset)
                self._groups.append(group)

    def extend(self, other, join="left"):
        """Extend InferenceData with groups from another InferenceData.

        Parameters
        ----------
        other : InferenceData
            InferenceData to be added
        join : {'left', 'right'}, default 'left'
            Defines how the two decide which group to keep when the same group is
            present in both objects. 'left' will discard the group in ``other`` whereas
            'right' will keep the group in ``other`` and discard the one in ``self``.

        See Also
        --------
        add_groups : Add new groups to InferenceData object.
        concat : Concatenate InferenceData objects.

        """
        if not isinstance(other, InferenceData):
            raise ValueError(
                "Extending is possible between two InferenceData objects only."
            )
        if join not in ("left", "right"):
            raise ValueError(
                "join must be either 'left' or 'right', found {}".format(join)
            )
        for group in other._groups_all:  # pylint: disable=protected-access
            if hasattr(self, group):
                if join == "left":
                    continue
            if group not in SUPPORTED_GROUPS:
                warnings.warn(
                    "{} group is not defined in the InferenceData scheme".format(group),
                    UserWarning,
                )
            dataset = getattr(other, group)
            setattr(self, group, dataset)
            self._groups.append(group)

    set_index = _extend_xr_method(xr.Dataset.set_index)
    get_index = _extend_xr_method(xr.Dataset.get_index)
    reset_index = _extend_xr_method(xr.Dataset.reset_index)
    set_coords = _extend_xr_method(xr.Dataset.set_coords)
    reset_coords = _extend_xr_method(xr.Dataset.reset_coords)
    assign = _extend_xr_method(xr.Dataset.assign)
    assign_coords = _extend_xr_method(xr.Dataset.assign_coords)
    sortby = _extend_xr_method(xr.Dataset.sortby)
    chunk = _extend_xr_method(xr.Dataset.chunk)
    unify_chunks = _extend_xr_method(xr.Dataset.unify_chunks)
    load = _extend_xr_method(xr.Dataset.load)
    compute = _extend_xr_method(xr.Dataset.compute)
    persist = _extend_xr_method(xr.Dataset.persist)

    mean = _extend_xr_method(xr.Dataset.mean)
    median = _extend_xr_method(xr.Dataset.median)
    min = _extend_xr_method(xr.Dataset.min)
    max = _extend_xr_method(xr.Dataset.max)
    cumsum = _extend_xr_method(xr.Dataset.cumsum)
    sum = _extend_xr_method(xr.Dataset.sum)
    quantile = _extend_xr_method(xr.Dataset.quantile)

    def _group_names(self, groups, filter_groups=None):
        """Handle expansion of group names input across arviz.

        Parameters
        ----------
        groups: str, list of str or None
            group or metagroup names.
        idata: xarray.Dataset
            Posterior data in an xarray
        filter_groups: {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup
            names. If "regex", interpret groups as regular expressions on the real group
            or metagroup names. A la `pandas.filter`.

        Returns
        -------
        groups: list
        """
        all_groups = self.groups
        if groups is None:
            return all_groups
        if isinstance(groups, str):
            groups = [groups]
        sel_groups = []
        metagroups = {"data": SUPPORTED_GROUPS}
        for group in groups:
            if group[0] == "~":
                sel_groups.extend(
                    [f"~{item}" for item in metagroups[group[1:]] if item in all_groups]
                    if group[1:] in metagroups
                    else [group]
                )
            else:
                sel_groups.extend(
                    [item for item in metagroups[group] if item in all_groups]
                    if group in metagroups
                    else [group]
                )

        try:
            group_names = _subset_list(
                sel_groups, all_groups, filter_items=filter_groups
            )
        except KeyError as err:
            msg = " ".join(("groups:", f"{err}", "in InferenceData"))
            raise KeyError(msg) from err
        return group_names

    def map(
        self, fun, groups=None, filter_groups=None, inplace=False, args=None, **kwargs
    ):
        """Apply a function to multiple groups.

        Applies ``fun`` groupwise to the selected ``InferenceData`` groups and
        overwrites the group with the result of the function.

        Parameters
        ----------
        fun : callable
            Function to be applied to each group. Assumes the function is called as
            ``fun(dataset, *args, **kwargs)``.
        groups : str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups : {None, "like", "regex"}, optional
            If `None` (default), interpret var_names as the real variables names. If
            "like", interpret var_names as substrings of the real variables names. If
            "regex", interpret var_names as regular expressions on the real variables
            names. A la `pandas.filter`.
        inplace : bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.
        args : array_like, optional
            Positional arguments passed to ``fun``.
        **kwargs : mapping, optional
            Keyword arguments passed to ``fun``.

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform selection in place and return `None`

        """
        if args is None:
            args = []
        groups = self._group_names(groups, filter_groups)

        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            dataset = fun(dataset, *args, **kwargs)
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def _wrap_xarray_method(
        self,
        method,
        groups=None,
        filter_groups=None,
        inplace=False,
        args=None,
        **kwargs,
    ):
        """Extend and xarray.Dataset method to InferenceData object.

        Parameters
        ----------
        method: str
            Method to be extended. Must be a ``xarray.Dataset`` method.
        groups: str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        inplace: bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.
        **kwargs: mapping, optional
            Keyword arguments passed to the xarray Dataset method.

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform selection in place and return `None`

        """
        if args is None:
            args = []
        groups = self._group_names(groups, filter_groups)

        method = getattr(xr.Dataset, method)

        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            dataset = method(dataset, *args, **kwargs)
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out


def _extend_xr_method(func):
    """Make wrapper to extend methods from xr.Dataset to InferenceData Class."""
    # pydocstyle requires a non empty line

    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        _filter = kwargs.pop("filter_groups", None)
        _groups = kwargs.pop("groups", None)
        _inplace = kwargs.pop("inplace", False)

        out = self if _inplace else deepcopy(self)

        groups = self._group_names(_groups, _filter)  # pylint: disable=protected-access
        for group in groups:
            xr_data = getattr(out, group)
            xr_data = func(xr_data, *args, **kwargs)  # pylint: disable=not-callable
            setattr(out, group, xr_data)

        return None if _inplace else out


def dict_to_dataset(
    data, *, attrs=None, library=None, coords=None, dims=None, skip_event_dims=None
):
    """Convert a dictionary of numpy arrays to an xarray.Dataset.
    Parameters
    ----------
    data : dict[str] -> ndarray
        Data to convert. Keys are variable names.
    attrs : dict
        Json serializable metadata to attach to the dataset, in addition to defaults.
    library : module
        Library used for performing inference. Will be attached to the attrs metadata.
    coords : dict[str] -> ndarray
        Coordinates for the dataset
    dims : dict[str] -> list[str]
        Dimensions of each variable. The keys are variable names, values are lists of
        coordinates.
    skip_event_dims : bool
        If True, cut extra dims whenever present to match the shape of the data.
        Necessary for PPLs which have the same name in both observed data and log
        likelihood groups, to account for their different shapes when observations are
        multivariate.
    Returns
    -------
    xr.Dataset
    Examples
    --------
    dict_to_dataset({'x': np.random.randn(4, 100), 'y': np.random.rand(4, 100)})
    """
    if dims is None:
        dims = {}

    data_vars = {}
    for key, values in data.items():
        data_vars[key] = numpy_to_data_array(
            values,
            var_name=key,
            coords=coords,
            dims=dims.get(key),
            skip_event_dims=skip_event_dims,
        )
    return xr.Dataset(data_vars=data_vars)


def numpy_to_data_array(
    ary, *, var_name="data", coords=None, dims=None, skip_event_dims=None
):
    """Convert a numpy array to an xarray.DataArray.
    The first two dimensions will be (chain, draw), and any remaining
    dimensions will be "shape".
    If the numpy array is 1d, this dimension is interpreted as draw
    If the numpy array is 2d, it is interpreted as (chain, draw)
    If the numpy array is 3 or more dimensions, the last dimensions are kept as shapes.
    Parameters
    ----------
    ary : np.ndarray
        A numpy array. If it has 2 or more dimensions, the first dimension should be
        independent chains from a simulation. Use `np.expand_dims(ary, 0)` to add a
        single dimension to the front if there is only 1 chain.
    var_name : str
        If there are no dims passed, this string is used to name dimensions
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : List(str)
        A list of coordinate names for the variable
    skip_event_dims : bool
    Returns
    -------
    xr.DataArray
        Will have the same data as passed, but with coordinates and dimensions
    """
    # manage and transform copies
    default_dims = ["index", "date"]
    ary = np.atleast_2d(ary)
    n_chains, n_samples, *shape = ary.shape
    if n_chains > n_samples:
        warnings.warn(
            "More chains ({n_chains}) than draws ({n_samples}). "
            "Passed array should have shape (chains, draws, *shape)".format(
                n_chains=n_chains, n_samples=n_samples
            ),
            UserWarning,
        )

    dims, coords = generate_dims_coords(
        shape,
        var_name,
        dims=dims,
        coords=coords,
        default_dims=default_dims,
        skip_event_dims=skip_event_dims,
    )

    # reversed order for default dims: 'index', 'date'
    if "date" not in dims:
        dims = ["date"] + dims
    if "index" not in dims:
        dims = ["index"] + dims

    if "index" not in coords:
        coords["index"] = np.arange(n_chains)
    if "date" not in coords:
        coords["date"] = np.arange(n_samples)

    # filter coords based on the dims
    coords = {key: xr.IndexVariable((key,), data=coords[key]) for key in dims}
    return xr.DataArray(ary, coords=coords, dims=dims)


def two_de(x):
    """Jitting numpy at_least_2d."""
    if not isinstance(x, np.ndarray):
        return np.atleast_2d(x)
    if x.ndim == 0:
        result = x.reshape(1, 1)
    elif x.ndim == 1:
        result = x[np.newaxis, :]
    else:
        result = x
    return result


class HtmlTemplate:
    """Contain html templates for InferenceData repr."""

    html_template = """
        <div>
          <div class='xr-header'>
            <div class="xr-obj-type">arviz.InferenceData</div>
          </div>
          <ul class="xr-sections group-sections">
          {}
          </ul>
        </div>
        """
    element_template = """
        <li class = "xr-section-item">
            <input id="idata_{group_id}" class="xr-section-summary-in" type="checkbox">
            <label for="idata_{group_id}" class = "xr-section-summary">{group}</label>
            <div class="xr-section-inline-details"></div>
            <div class="xr-section-details">
                <ul id="xr-dataset-coord-list" class="xr-var-list">
                    <div style="padding-left:2rem;">{xr_data}<br></div>
                </ul>
            </div>
        </li>
        """
    # _, css_style = _load_static_files()  # pylint: disable=protected-access
    specific_style = ".xr-wrap{width:700px!important;}"
    css_template = f"<style> {specific_style} </style>"


def either_dict_or_kwargs(
    pos_kwargs,
    kw_kwargs,
    func_name,
):
    """Clone from xarray.core.utils."""
    if pos_kwargs is not None:
        if not hasattr(pos_kwargs, "keys") and hasattr(pos_kwargs, "__getitem__"):
            raise ValueError(
                "the first argument to .%s must be a dictionary" % func_name
            )
        if kw_kwargs:
            raise ValueError(
                "cannot specify both keyword and positional "
                "arguments to .%s" % func_name
            )
        return pos_kwargs
    else:
        return kw_kwargs


def _subset_list(subset, whole_list, filter_items=None, warn=True):
    """Handle list subsetting (var_names, groups...) across arviz.
    Parameters
    ----------
    subset : str, list, or None
    whole_list : list
        List from which to select a subset according to subset elements and
        filter_items value.
    filter_items : {None, "like", "regex"}, optional
        If `None` (default), interpret `subset` as the exact elements in `whole_list`
        names. If "like", interpret `subset` as substrings of the elements in
        `whole_list`. If "regex", interpret `subset` as regular expressions to match
        elements in `whole_list`. A la `pandas.filter`.
    Returns
    -------
    list or None
        A subset of ``whole_list`` fulfilling the requests imposed by ``subset``
        and ``filter_items``.
    """
    if subset is not None:

        if isinstance(subset, str):
            subset = [subset]

        whole_list_tilde = [item for item in whole_list if item.startswith("~")]
        if whole_list_tilde and warn:
            warnings.warn(
                "ArviZ treats '~' as a negation character for selection. There are "
                "elements in `whole_list` starting with '~', {0}. Please double check"
                "your results to ensure all elements are included".format(
                    ", ".join(whole_list_tilde)
                )
            )

        excluded_items = [
            item[1:]
            for item in subset
            if item.startswith("~") and item not in whole_list
        ]
        filter_items = str(filter_items).lower()
        not_found = []

        if excluded_items:
            if filter_items in ("like", "regex"):
                for pattern in excluded_items[:]:
                    excluded_items.remove(pattern)
                    if filter_items == "like":
                        real_items = [
                            real_item
                            for real_item in whole_list
                            if pattern in real_item
                        ]
                    else:
                        # i.e filter_items == "regex"
                        real_items = [
                            real_item
                            for real_item in whole_list
                            if re.search(pattern, real_item)
                        ]
                    if not real_items:
                        not_found.append(pattern)
                    excluded_items.extend(real_items)
            not_found.extend(
                [item for item in excluded_items if item not in whole_list]
            )
            if not_found:
                warnings.warn(
                    f"Items starting with ~: {not_found} have not been found and will "
                    "be ignored"
                )
            subset = [item for item in whole_list if item not in excluded_items]

        else:
            if filter_items == "like":
                subset = [
                    item for item in whole_list for name in subset if name in item
                ]
            elif filter_items == "regex":
                subset = [
                    item
                    for item in whole_list
                    for name in subset
                    if re.search(name, item)
                ]

        existing_items = np.isin(subset, whole_list)
        if not np.all(existing_items):
            raise KeyError(
                "{} are not present".format(np.array(subset)[~existing_items])
            )

    return subset


def generate_dims_coords(
    shape, var_name, dims=None, coords=None, default_dims=None, skip_event_dims=None
):
    """Generate default dimensions and coordinates for a variable.
    Parameters
    ----------
    shape : tuple[int]
        Shape of the variable
    var_name : str
        Name of the variable. If no dimension name(s) is provided, ArviZ
        will generate a default dimension name using ``var_name``, e.g.,
        ``"foo_dim_0"`` for the first dimension if ``var_name`` is ``"foo"``.
    dims : list
        List of dimensions for the variable
    coords : dict[str] -> list[str]
        Map of dimensions to coordinates
    default_dims : list[str]
        Dimension names that are not part of the variable's shape. For example,
        when manipulating Monte Carlo traces, the ``default_dims`` would be
        ``["chain" , "draw"]`` which ArviZ uses as its own names for dimensions
        of MCMC traces.
    skip_event_dims : bool, default False
    Returns
    -------
    list[str]
        Default dims
    dict[str] -> list[str]
        Default coords
    """
    if default_dims is None:
        default_dims = []
    if dims is None:
        dims = []
    if skip_event_dims is None:
        skip_event_dims = False

    if coords is None:
        coords = {}

    coords = deepcopy(coords)
    dims = deepcopy(dims)

    ndims = len([dim for dim in dims if dim not in default_dims])
    if ndims > len(shape):
        if skip_event_dims:
            dims = dims[: len(shape)]
        else:
            warnings.warn(
                (
                    "In variable {var_name}, there are "
                    + "more dims ({dims_len}) given than exist ({shape_len}). "
                    + "Passed array should have shape ({defaults}*shape)"
                ).format(
                    var_name=var_name,
                    dims_len=len(dims),
                    shape_len=len(shape),
                    defaults=",".join(default_dims) + ", "
                    if default_dims is not None
                    else "",
                ),
                UserWarning,
            )
    if skip_event_dims:
        # this is needed in case the reduction keeps the dimension with size 1
        for i, (dim, dim_size) in enumerate(zip(dims, shape)):
            print(f"{i}, dim: {dim}, {dim_size} =? {len(coords.get(dim, []))}")
            if (dim in coords) and (dim_size != len(coords[dim])):
                dims = dims[:i]
                break

    for idx, dim_len in enumerate(shape):
        if (len(dims) < idx + 1) or (dims[idx] is None):
            dim_name = "{var_name}_dim_{idx}".format(var_name=var_name, idx=idx)
            if len(dims) < idx + 1:
                dims.append(dim_name)
            else:
                dims[idx] = dim_name
        dim_name = dims[idx]
        if dim_name not in coords:
            coords[dim_name] = np.arange(dim_len)
    coords = {
        key: coord for key, coord in coords.items() if any(key == dim for dim in dims)
    }
    return dims, coords
